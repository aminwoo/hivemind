#include "interface/uci.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <memory>

#include "nn/onnx_utils.h"
#include "environment/planes.h"
#include "common/utils.h"
#include "common/globals.h"

using namespace std;

namespace {

bool same_action(const JointActionCandidate& lhs,
                 const JointActionCandidate& rhs) {
    return lhs.moveA == rhs.moveA && lhs.moveB == rhs.moveB;
}

string format_joint_action(Board& board,
                           const JointActionCandidate& action) {
    const string moveA = action.moveA == Stockfish::MOVE_NONE
        ? "pass" : board.uci_move(BOARD_A, action.moveA);
    const string moveB = action.moveB == Stockfish::MOVE_NONE
        ? "pass" : board.uci_move(BOARD_B, action.moveB);
    return "(" + moveA + "," + moveB + ")";
}

}  // namespace

MergedRootResult merge_root_results(const vector<RootTreeResult>& trees) {
    struct Accumulator {
        RootEdgeStats edge;
        double qWeighted = 0.0;
        int qWeight = 0;
        int proofVotes = 0;
    };

    vector<Accumulator> merged;
    for (const RootTreeResult& tree : trees) {
        for (const RootEdgeStats& edge : tree.edges) {
            auto found = find_if(merged.begin(), merged.end(), [&](const auto& item) {
                return same_action(item.edge.action, edge.action);
            });
            if (found == merged.end()) {
                merged.push_back({});
                found = prev(merged.end());
                found->edge.action = edge.action;
            }
            found->edge.visits += edge.visits;
            const int weight = max(1, edge.visits);
            found->qWeighted += static_cast<double>(edge.q) * weight;
            found->qWeight += weight;
        }
    }

    // A solver proof is authoritative. In a healthy search all trees agree on
    // the root type, but preferring a proof over an unfinished tree also makes
    // early exits useful when only one GPU reaches the terminal line in time.
    NodeType rootType = NodeType::UNSOLVED;
    for (NodeType candidate : {NodeType::WIN, NodeType::LOSS, NodeType::DRAW}) {
        if (any_of(trees.begin(), trees.end(), [&](const RootTreeResult& tree) {
                return tree.rootType == candidate;
            })) {
            rootType = candidate;
            break;
        }
    }
    if (rootType != NodeType::UNSOLVED) {
        for (const RootTreeResult& tree : trees) {
            if (tree.rootType != rootType || tree.edges.empty()) {
                continue;
            }
            auto found = find_if(merged.begin(), merged.end(), [&](const auto& item) {
                return same_action(item.edge.action, tree.decision);
            });
            if (found != merged.end()) {
                ++found->proofVotes;
            }
        }
    }

    for (Accumulator& item : merged) {
        item.edge.q = item.qWeight > 0
            ? static_cast<float>(item.qWeighted / item.qWeight)
            : 0.0f;
    }
    sort(merged.begin(), merged.end(), [&](const Accumulator& lhs,
                                           const Accumulator& rhs) {
        if (rootType != NodeType::UNSOLVED
            && lhs.proofVotes != rhs.proofVotes) {
            return lhs.proofVotes > rhs.proofVotes;
        }
        if (lhs.edge.visits != rhs.edge.visits) {
            return lhs.edge.visits > rhs.edge.visits;
        }
        return lhs.edge.q > rhs.edge.q;
    });

    MergedRootResult result;
    result.rootType = rootType;
    result.edges.reserve(merged.size());
    for (const Accumulator& item : merged) {
        result.edges.push_back(item.edge);
    }
    if (result.edges.empty()) {
        return result;
    }
    result.hasAction = true;
    result.action = result.edges.front().action;

    int representativeVisits = -1;
    for (size_t treeIndex = 0; treeIndex < trees.size(); ++treeIndex) {
        const RootTreeResult& tree = trees[treeIndex];
        if (!same_action(tree.decision, result.action)) {
            continue;
        }
        const auto edge = find_if(tree.edges.begin(), tree.edges.end(), [&](const auto& item) {
            return same_action(item.action, result.action);
        });
        const int visits = edge == tree.edges.end() ? 0 : edge->visits;
        if (visits > representativeVisits) {
            representativeVisits = visits;
            result.representativeTree = treeIndex;
        }
    }
    return result;
}

UCI::UCI() : mainSearchThread(nullptr) {

}

UCI::~UCI() {
    stop();
}

bool UCI::initializeEngines(
    const std::vector<int>& deviceIdsToUse,
    const std::string& networkPathToUse,
    int batchSizeToUse) {
    deviceIds = deviceIdsToUse;
    networkPath = networkPathToUse;
    batchSize = batchSizeToUse > 0 ? batchSizeToUse : SearchParams::BATCH_SIZE;
    return reload_engines();
}

bool UCI::reload_engines() {
    stop();

    // Clear any existing engines.
    engines.clear();

    const std::string onnxFile = resolveModelPath(networkPath);
    if (onnxFile.empty()) {
        std::cerr << "Error: No ONNX model found; pass --model <onnx> or --network <onnx>." << std::endl;
        return false;
    }
    if (!std::filesystem::is_regular_file(onnxFile)) {
        std::cerr << "Error: ONNX model not found: " << onnxFile << std::endl;
        return false;
    }
    // For each device ID, create a new Engine, load the network, and store it.
    for (int deviceId : deviceIds) {
        const std::string engineFile = getEnginePath(onnxFile, "fp16", batchSize, deviceId, "v3");

        // Create a new engine instance on the given GPU.
        auto enginePtr = std::make_unique<Engine>(deviceId, batchSize);

        // Attempt to load the network (build or deserialize).
        if (!enginePtr->loadNetwork(onnxFile, engineFile)) {
            std::cerr << "Error: Failed to load engine on device " << deviceId << std::endl;
        } else {
            engines.push_back(std::move(enginePtr));
        }
    }

    agents.clear();
    ensure_agents(required_agent_count());
    return !engines.empty();
}


size_t UCI::required_agent_count() const {
    if (engines.empty()) {
        return 0;
    }
    return rootParallelism ? engines.size() : 1;
}

void UCI::ensure_agents(size_t count) {
    if (agents.size() > count) {
        agents.resize(count);
        return;
    }
    while (agents.size() < count) {
        agents.push_back(std::make_unique<Agent>());
        if (hashSizeMB > 0) {
            agents.back()->setHashSize(hashSizeMB);
        }
    }
}

void UCI::stop() {
    for (const auto& agent : agents) {
        agent->set_is_running(false);
    }
    if (mainSearchThread) {
        if (mainSearchThread->joinable()) {
            mainSearchThread->join();
        }
        delete mainSearchThread;
        mainSearchThread = nullptr;
    }
    ongoingSearch.store(false, std::memory_order_release);
}

void UCI::new_game() {
    stop();
    for (const auto& agent : agents) {
        agent->reset_search_state();
    }
}

void UCI::position(istringstream& is) {
    stop();

    std::string token;
    is >> token;
    
    // Set the board position
    if (token == "startpos") {
        // Use a predefined starting FEN for the initial position.
        board.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1|rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }
    else if (token == "fen") {
        // Build the FEN string until we hit "moves" or end of stream
        std::string fen;
        while (is >> token && token != "moves") {
            fen += token + " ";
        }
        board.set(fen);
        
        if (token == "moves") {
            is.seekg(-6, std::ios_base::cur);  
        }
    }
    else {
        return;
    }
    
    if (is >> token && token == "moves") {
        // Parse move list (if any)
        int moveCount = 0;
        while (is >> token) {
            if (token.empty() || token[0] < '1' || token[0] > '2') {
                std::cerr << "Error: Invalid board indicator in move '" << token 
                          << "' at move " << (moveCount + 1) << std::endl;
                break;
            }
            int boardNum = token[0] - '1'; // '1' becomes 0, '2' becomes 1.
            std::string moveStr = token.substr(1); // Extract move string without board indicator
            Stockfish::Move m = Stockfish::UCI::to_move(*board.pos[boardNum], moveStr);
            if (m == Stockfish::MOVE_NONE) {
                std::cerr << "Error: Invalid move '" << moveStr << "' on board " 
                          << (boardNum + 1) << " at move " << (moveCount + 1) << std::endl;
                std::cerr << "       Current FEN: " << board.fen(boardNum) << std::endl;
                std::cerr << "       Legal moves: ";
                auto legalMoves = board.legal_moves(boardNum);
                for (const auto& lm : legalMoves) {
                    std::cerr << board.uci_move(boardNum, lm) << " ";
                }
                std::cerr << std::endl;
                break;  // Stop if an invalid move is encountered.
            }
            board.push_move(boardNum, m);
            moveCount++;
        }
    }
}

JointActionCandidate UCI::run_root_parallel_search(
    const SearchOptions& options) {
    const size_t treeCount = engines.size();
    vector<RootTreeResult> trees(treeCount);
    vector<SearchRunStats> stats(treeCount);
    vector<exception_ptr> failures(treeCount);
    vector<thread> treeThreads;
    treeThreads.reserve(treeCount);

    for (size_t index = 0; index < treeCount; ++index) {
        treeThreads.emplace_back([&, index] {
            try {
                Board localBoard(board);
                SearchOptions localOptions = options;
                localOptions.verbose = false;
                localOptions.enablePonder = false;
                if (localOptions.targetNodes > 0) {
                    localOptions.targetNodes = max<size_t>(
                        1, (localOptions.targetNodes + treeCount - 1) / treeCount);
                }
                vector<Engine*> localEngines = {engines[index].get()};
                trees[index].decision = agents[index]->run_search(
                    localBoard, localEngines, teamSide,
                    teamHasTimeAdvantage, localOptions);
                trees[index].rootType = agents[index]->root_type();
                trees[index].edges = agents[index]->root_edge_stats();
                stats[index] = agents[index]->last_search_stats();
            } catch (...) {
                failures[index] = current_exception();
            }
        });
    }
    for (thread& worker : treeThreads) {
        worker.join();
    }
    for (const exception_ptr& failure : failures) {
        if (failure) {
            rethrow_exception(failure);
        }
    }

    const MergedRootResult merged = merge_root_results(trees);
    if (!merged.hasAction) {
        cout << "bestmove (none)" << endl;
        return {};
    }

    int totalNodes = 0;
    int maxDepth = 0;
    int elapsedMs = 0;
    int sameBatchCollisions = 0;
    int reservationCollisions = 0;
    for (const SearchRunStats& treeStats : stats) {
        totalNodes += treeStats.nodes;
        maxDepth = max(maxDepth, treeStats.depth);
        elapsedMs = max(elapsedMs, treeStats.elapsedMs);
        sameBatchCollisions += treeStats.sameBatchCollisions;
        reservationCollisions += treeStats.reservationCollisions;
    }
    elapsedMs = max(1, elapsedMs);
    const int nps = static_cast<int>(
        static_cast<double>(totalNodes) * 1000.0 / elapsedMs);

    const int pvCount = min<int>(options.multiPV, merged.edges.size());
    for (int pvIndex = 0; pvIndex < pvCount; ++pvIndex) {
        const RootEdgeStats& edge = merged.edges[pvIndex];
        cout << "info depth " << maxDepth;
        if (options.multiPV > 1) {
            cout << " multipv " << pvIndex + 1;
        }
        if (pvIndex == 0 && merged.rootType != NodeType::UNSOLVED) {
            const int mate = max(1, (stats[merged.representativeTree].depth + 1) / 2);
            if (merged.rootType == NodeType::WIN) {
                cout << " score mate " << mate;
            } else if (merged.rootType == NodeType::LOSS) {
                cout << " score mate -" << mate;
            } else {
                cout << " score cp 0";
            }
        } else {
            const int cp = static_cast<int>(
                180.0f * tan(1.56f * clamp(edge.q, -0.999f, 0.999f)));
            cout << " score cp " << cp;
        }
        cout << " nodes " << totalNodes
             << " nps " << nps
             << " hashfull 0 tbhits 0"
             << " time " << elapsedMs
             << " pv " << format_joint_action(board, edge.action)
             << endl;
    }
    cout << "info string root parallel trees " << treeCount
         << " workers " << treeCount * SearchParams::NUM_SEARCH_THREADS
         << endl;
    cout << "info string rejected selection attempts "
         << sameBatchCollisions + reservationCollisions
         << " (same batch " << sameBatchCollisions
         << ", pending evaluation " << reservationCollisions << ")"
         << " per 1000 nodes "
         << (totalNodes > 0
                 ? 1000.0 * (sameBatchCollisions + reservationCollisions)
                     / static_cast<double>(totalNodes)
                 : 0.0)
         << endl;

    // Per-tree root distributions. Root parallelism is only worth its cost if
    // the trees search differently; with identical priors, no root noise and a
    // shared seed they can converge on the same lines and the merge then
    // averages four copies of one search. Reporting each tree's own top edges
    // makes that visible rather than assumed.
    for (size_t index = 0; index < treeCount; ++index) {
        vector<RootEdgeStats> edges = trees[index].edges;
        sort(edges.begin(), edges.end(),
             [](const RootEdgeStats& lhs, const RootEdgeStats& rhs) {
                 return lhs.visits > rhs.visits;
             });
        int treeVisits = 0;
        for (const RootEdgeStats& edge : edges) {
            treeVisits += edge.visits;
        }
        cout << "info string tree " << index
             << " nodes " << stats[index].nodes
             << " depth " << stats[index].depth
             << " visits " << treeVisits
             << " top";
        const size_t shown = min<size_t>(5, edges.size());
        for (size_t rank = 0; rank < shown; ++rank) {
            cout << " " << format_joint_action(board, edges[rank].action)
                 << ":" << edges[rank].visits;
        }
        cout << endl;
    }

    const string bestMove = format_joint_action(board, merged.action);
    string ponderMove;
    if (options.enablePonder
        && merged.representativeTree < agents.size()
        && same_action(trees[merged.representativeTree].decision, merged.action)) {
        ponderMove = agents[merged.representativeTree]->extract_ponder_move(board);
    }
    cout << "bestmove " << bestMove;
    if (!ponderMove.empty()) {
        cout << " ponder " << ponderMove;
    }
    cout << endl;
    return merged.action;
}

void UCI::run_root_parallel_brain(const JointActionCandidate& played,
                                  const SearchOptions& options) {
    vector<thread> treeThreads;
    treeThreads.reserve(engines.size());
    for (size_t index = 0; index < engines.size(); ++index) {
        treeThreads.emplace_back([&, index] {
            Board localBoard(board);
            vector<Engine*> localEngines = {engines[index].get()};
            agents[index]->run_permanent_brain(
                localBoard, localEngines, teamSide,
                teamHasTimeAdvantage, played, options);
        });
    }
    for (thread& worker : treeThreads) {
        worker.join();
    }
}

void UCI::go(std::istringstream& is) {
    std::string token;
    int moveTime = 0;
    size_t nodes = 0;
    bool isPonder = false;
    
    // Parse go parameters
    while (is >> token) {
        if (token == "ponder") {
            isPonder = true;
        } else if (token == "movetime") {
            is >> moveTime;
        } else if (token == "nodes") {
            is >> nodes;
        }
    }
    
    stop();

    // Ensure that engines have been initialized.
    if (agents.empty() || engines.empty()) {
        std::cerr << "Error: No engines have been initialized!" << std::endl;
        return;
    }

    ongoingSearch.store(true, std::memory_order_release);

    // Build a vector of raw Engine pointers from the unique_ptr collection.
    std::vector<Engine*> enginePtrs;
    enginePtrs.reserve(engines.size());
    for (const auto& eng : engines) {
        enginePtrs.push_back(eng.get());
    }

    // Build search options based on what was specified
    SearchOptions opts;
    if (nodes > 0) {
        opts = SearchOptions::uci(static_cast<int>(nodes), multiPV, isPonder);
        opts.moveTimeMs = 0;  // Node-based search
        opts.targetNodes = nodes;
    } else if (moveTime > 0) {
        opts = SearchOptions::uci(moveTime, multiPV, isPonder);
    } else {
        // Default to 1 second if nothing specified
        opts = SearchOptions::uci(1000, multiPV, isPonder);
    }
    opts.enablePonder = ponderEnabled;
    opts.search = searchConfig;
    const bool useRootParallelism = rootParallelism && engines.size() > 1;
    // stop() above joined any previous search, so resizing here cannot destroy
    // an agent that is still working.
    ensure_agents(useRootParallelism ? engines.size() : 1);
    
    // Launch the search thread
    mainSearchThread = new std::thread(
        [this, enginePtrs, opts, useRootParallelism]() {
        try {
            const JointActionCandidate played = useRootParallelism
                ? run_root_parallel_search(opts)
                : agents.front()->run_search(
                    board, enginePtrs, teamSide, teamHasTimeAdvantage, opts);
            if (!opts.isPonder) {
                // Nothing else runs between moves. Keep thinking from the
                // position our own move creates, so whichever reply the four
                // asynchronous players produce lands on a subtree that already
                // has work in it. Returns on the next position, go, or stop.
                if (useRootParallelism) {
                    run_root_parallel_brain(played, opts);
                } else {
                    agents.front()->run_permanent_brain(
                        board, enginePtrs, teamSide, teamHasTimeAdvantage,
                        played, opts);
                }
            }
        } catch (const std::exception& error) {
            std::cerr << "Search failed: " << error.what() << std::endl;
            std::cout << "info string search failed: " << error.what() << std::endl;
            std::cout << "bestmove (none)" << std::endl;
        } catch (...) {
            std::cerr << "Search failed with an unknown exception" << std::endl;
            std::cout << "info string search failed: unknown exception" << std::endl;
            std::cout << "bestmove (none)" << std::endl;
        }
        ongoingSearch.store(false, std::memory_order_release);
    });

    while (ongoingSearch.load(std::memory_order_acquire)
           && none_of(agents.begin(), agents.end(), [](const auto& agent) {
               return agent->is_running();
           })) {
        std::this_thread::yield();
    }
}

void UCI::ponderhit() {
    if (ongoingSearch.load(std::memory_order_acquire)) {
        for (const auto& agent : agents) {
            agent->ponderhit();
        }
    }
}

void UCI::setoption(std::istringstream& is) {
    stop();

    std::string token;
    is >> token; 
    if (token != "name") return;
    std::string name;
    is >> name;
    is >> token; 
    if (token != "value") return;
    std::string value;
    is >> value;
    if (name == "Hash") {
        // Parse hash size in MB (1 - 33554432 MB)
        size_t sizeMB = std::stoull(value);
        
        // Each root tree owns a transposition table, so this is a per-tree
        // size: sharing one graph allocates it once however many GPUs drive it.
        hashSizeMB = sizeMB;
        for (const auto& agent : agents) {
            agent->setHashSize(sizeMB);
        }
        std::cout << "info string Hash table set to " << sizeMB
                  << " MB per search tree" << std::endl;
    } else if (name == "BatchSize") {
        // The batch size is compiled into the TensorRT engine, so this reloads
        // it. A cached engine for the requested size loads in about a second;
        // an uncached one is built from the ONNX first, which takes minutes.
        const int requested = std::stoi(value);
        if (requested < 1 || requested > 1024) {
            std::cout << "info string BatchSize must be between 1 and 1024" << std::endl;
        } else if (requested == batchSize) {
            std::cout << "info string BatchSize already " << batchSize << std::endl;
        } else {
            const int previous = batchSize;
            batchSize = requested;
            if (reload_engines()) {
                std::cout << "info string BatchSize set to " << batchSize << std::endl;
            } else {
                batchSize = previous;
                std::cout << "info string BatchSize " << requested
                          << " failed to load; restoring " << previous << std::endl;
                if (!reload_engines()) {
                    std::cout << "info string no inference engine is loaded" << std::endl;
                }
            }
        }
    } else if (name == "MultiPV") {
        int mpv = std::stoi(value);
        if (mpv >= 1 && mpv <= 500) {
            multiPV = mpv;
            std::cout << "info string MultiPV set to " << multiPV << std::endl;
        }
    } else if (name == "Ponder") {
        if (value == "true" || value == "false") {
            ponderEnabled = (value == "true");
            std::cout << "info string Ponder set to " << value << std::endl;
        }
    } else if (name == "RootParallelism") {
        if (value == "true" || value == "false") {
            rootParallelism = value == "true";
            for (const auto& agent : agents) {
                agent->reset_search_state();
            }
            std::cout << "info string RootParallelism set to " << value
                      << std::endl;
        }
    } else if (name == "DrawContemptPermille") {
        int permille = std::clamp(std::stoi(value), 0, 1000);
        searchConfig.drawContempt = static_cast<float>(permille) / 1000.0f;
        std::cout << "info string DrawContemptPermille set to " << permille << std::endl;
    } else if (name == "MovesLeftDiscountPermille") {
        const int permille = std::clamp(std::stoi(value), 0, 1000);
        searchConfig.movesLeftDiscount = static_cast<float>(permille) / 1000.0f;
        std::cout << "info string MovesLeftDiscountPermille set to " << permille
                  << std::endl;
    } else if (name == "PWCoefficientPermille") {
        int permille = std::clamp(std::stoi(value), 1, 10000);
        searchConfig.pwCoefficient = static_cast<float>(permille) / 1000.0f;
        std::cout << "info string PWCoefficientPermille set to " << permille << std::endl;
    } else if (name == "RootPWCoefficientPermille") {
        int permille = std::clamp(std::stoi(value), 1, 10000);
        searchConfig.rootPwCoefficient = static_cast<float>(permille) / 1000.0f;
        std::cout << "info string RootPWCoefficientPermille set to " << permille << std::endl;
    } else if (name == "PWExponentPermille") {
        int permille = std::clamp(std::stoi(value), 1, 1000);
        searchConfig.pwExponent = static_cast<float>(permille) / 1000.0f;
        std::cout << "info string PWExponentPermille set to " << permille << std::endl;
    } else if (name == "Transpositions") {
        if (value == "true" || value == "false") {
            searchConfig.enableTranspositions = value == "true";
            std::cout << "info string Transpositions set to " << value << std::endl;
        }
    } else if (name == "GumbelRootSearch") {
        if (value == "true" || value == "false") {
            searchConfig.enableGumbelRootSearch = value == "true";
            std::cout << "info string GumbelRootSearch set to " << value << std::endl;
        }
    } else if (name == "RootGumbelPoolSize") {
        searchConfig.rootGumbelPoolSize = std::clamp(
            std::stoi(value), 1, 65536);
        std::cout << "info string RootGumbelPoolSize set to "
                  << searchConfig.rootGumbelPoolSize << std::endl;
    } else if (name == "RootGumbelInitialCandidates") {
        searchConfig.rootGumbelInitialCandidates = std::clamp(
            std::stoi(value), 1, 4096);
        std::cout << "info string RootGumbelInitialCandidates set to "
                  << searchConfig.rootGumbelInitialCandidates << std::endl;
    } else if (name == "RootGumbelReplenishment") {
        searchConfig.rootGumbelReplenishment = std::clamp(
            std::stoi(value), 1, 4096);
        std::cout << "info string RootGumbelReplenishment set to "
                  << searchConfig.rootGumbelReplenishment << std::endl;
    } else if (name == "RootGumbelValueScalePermille") {
        const int permille = std::clamp(std::stoi(value), 0, 10000);
        searchConfig.rootGumbelValueScale =
            static_cast<float>(permille) / 1000.0f;
        std::cout << "info string RootGumbelValueScalePermille set to "
                  << permille << std::endl;
    } else if (name == "RootGumbelMaxRoundVisits") {
        searchConfig.rootGumbelMaxRoundVisits = std::clamp(
            std::stoi(value), 1, 100000);
        std::cout << "info string RootGumbelMaxRoundVisits set to "
                  << searchConfig.rootGumbelMaxRoundVisits << std::endl;
    } else if (name == "Team") {
        if (value == "white") {
            teamSide = Stockfish::WHITE;
        } else if (value == "black") {
            teamSide = Stockfish::BLACK;
        }
    } else if (name == "TimeAdvantage") {
        if (value == "true" || value == "false") {
            teamHasTimeAdvantage = value == "true";
            std::cout << "info string TimeAdvantage set to " << value << std::endl;
        }
    } else if (name == "Mode") {
        // Deprecated alias for TimeAdvantage, kept so existing GUI profiles keep
        // working. "sit" meant the team was up on time and could wait; "go"
        // meant it was not. Not advertised in the uci handshake.
        if (value == "sit" || value == "go") {
            teamHasTimeAdvantage = value == "sit";
            std::cout << "info string Mode is deprecated; use "
                      << "'setoption name TimeAdvantage value "
                      << (teamHasTimeAdvantage ? "true" : "false") << "'"
                      << std::endl;
        }
    }
}

void UCI::send_uci_response() {
    cout << "id name hivemind" << endl;
    cout << "id author aminwoo\n" << endl;
    cout << "option name Hash type spin default 16 min 1 max 33554432" << endl;
    cout << "option name BatchSize type spin default " << batchSize
         << " min 1 max 1024" << endl;
    cout << "option name MultiPV type spin default 1 min 1 max 500" << endl;
    cout << "option name Ponder type check default true" << endl;
    cout << "option name RootParallelism type check default false" << endl;
    cout << "option name DrawContemptPermille type spin default 0 min 0 max 1000" << endl;
    cout << "option name MovesLeftDiscountPermille type spin default "
         << static_cast<int>(SearchParams::MOVES_LEFT_DISCOUNT * 1000.0f)
         << " min 0 max 1000" << endl;
    cout << "option name PWCoefficientPermille type spin default "
         << static_cast<int>(SearchParams::PW_COEFFICIENT * 1000.0f) << " min 1 max 10000" << endl;
    cout << "option name RootPWCoefficientPermille type spin default "
         << static_cast<int>(SearchParams::ROOT_PW_COEFFICIENT * 1000.0f) << " min 1 max 10000" << endl;
    cout << "option name PWExponentPermille type spin default "
         << static_cast<int>(SearchParams::PW_EXPONENT * 1000.0f) << " min 1 max 1000" << endl;
    cout << "option name Transpositions type check default "
        << (SearchParams::ENABLE_TRANSPOSITIONS ? "true" : "false") << endl;
    cout << "option name GumbelRootSearch type check default "
         << (SearchParams::ENABLE_GUMBEL_ROOT_SEARCH ? "true" : "false") << endl;
    cout << "option name RootGumbelPoolSize type spin default "
         << SearchParams::ROOT_GUMBEL_POOL_SIZE << " min 1 max 65536" << endl;
    cout << "option name RootGumbelInitialCandidates type spin default "
         << SearchParams::ROOT_GUMBEL_INITIAL_CANDIDATES
         << " min 1 max 4096" << endl;
    cout << "option name RootGumbelReplenishment type spin default "
         << SearchParams::ROOT_GUMBEL_REPLENISHMENT
         << " min 1 max 4096" << endl;
    cout << "option name RootGumbelValueScalePermille type spin default "
         << static_cast<int>(SearchParams::ROOT_GUMBEL_VALUE_SCALE * 1000.0f)
         << " min 0 max 10000" << endl;
    cout << "option name RootGumbelMaxRoundVisits type spin default "
         << SearchParams::ROOT_GUMBEL_MAX_ROUND_VISITS
         << " min 1 max 100000" << endl;
    cout << "option name Team type combo default white var white var black" << endl;
    cout << "option name TimeAdvantage type check default false" << endl;
    cout << "info string CUDA engines " << engines.size()
         << " search workers " << engines.size() * SearchParams::NUM_SEARCH_THREADS
         << " (" << SearchParams::NUM_SEARCH_THREADS << " per engine)"
         << " root trees " << (rootParallelism ? engines.size() : 1) << endl;
    cout << "uciok" << endl;
}

void UCI::policy() {
    stop();
    if (engines.empty()) {
        cerr << "Error: No engines have been initialized!" << endl;
        return;
    }

    // Allocate inference buffers. runInference reads and writes a full batch,
    // so these must be sized by the loaded engine, not by the compiled default.
    const size_t loadedBatchSize =
        static_cast<size_t>(engines[0]->getBatchSize());
    float* obs = new float[loadedBatchSize * NB_INPUT_VALUES()];
    float* value = new float[loadedBatchSize];
    float* piA = new float[loadedBatchSize * NB_POLICY_VALUES()];
    float* piB = new float[loadedBatchSize * NB_POLICY_VALUES()];
    float* wdl = new float[loadedBatchSize * 3];
    float* movesLeft = new float[loadedBatchSize];

    // Convert board to planes
    board_to_planes(board, obs, teamSide, teamHasTimeAdvantage);

    // Run inference
    Engine* engine = engines[0].get();
    if (!engine->runInference(obs, value, piA, piB, wdl, movesLeft)) {
        cerr << "Inference failed" << endl;
        delete[] obs;
        delete[] value;
        delete[] piA;
        delete[] piB;
        delete[] wdl;
        delete[] movesLeft;
        return;
    }

    cout << "Value: " << value[0] << endl;
        const float maxWdl = std::max({wdl[0], wdl[1], wdl[2]});
        const float lossExp = std::exp(wdl[0] - maxWdl);
        const float drawExp = std::exp(wdl[1] - maxWdl);
        const float winExp = std::exp(wdl[2] - maxWdl);
        const float wdlTotal = lossExp + drawExp + winExp;
        cout << "WDL: " << winExp / wdlTotal << " " << drawExp / wdlTotal
            << " " << lossExp / wdlTotal << endl;
        cout << "Predicted plies to end: " << movesLeft[0] * 100.0f << endl;
    cout << endl;

    // Board A policy
    cout << "Board A (" << board.fen(BOARD_A) << "):" << endl;
    if (board.side_to_move(BOARD_A) == teamSide) {
        vector<Stockfish::Move> actionsA = board.legal_moves(BOARD_A);
        actionsA.push_back(Stockfish::MOVE_NONE);  // Add sit option
        vector<float> priorsA = get_normalized_probability(
            piA, actionsA, BOARD_A, board);
        
        // Sort by probability (descending)
        vector<size_t> indices = argsort(priorsA);
        for (size_t idx : indices) {
            string moveStr = (actionsA[idx] == Stockfish::MOVE_NONE) 
                            ? "pass" : board.uci_move(BOARD_A, actionsA[idx]);
            cout << "  " << moveStr << ": " << priorsA[idx] << endl;
        }
    } else {
        cout << "  (not our turn)" << endl;
    }
    cout << endl;

    // Board B policy
    cout << "Board B (" << board.fen(BOARD_B) << "):" << endl;
    if (board.side_to_move(BOARD_B) == ~teamSide) {
        vector<Stockfish::Move> actionsB = board.legal_moves(BOARD_B);
        actionsB.push_back(Stockfish::MOVE_NONE);  // Add sit option
        vector<float> priorsB = get_normalized_probability(
            piB, actionsB, BOARD_B, board);
        
        // Sort by probability (descending)
        vector<size_t> indices = argsort(priorsB);
        for (size_t idx : indices) {
            string moveStr = (actionsB[idx] == Stockfish::MOVE_NONE) 
                            ? "pass" : board.uci_move(BOARD_B, actionsB[idx]);
            cout << "  " << moveStr << ": " << priorsB[idx] << endl;
        }
    } else {
        cout << "  (not our turn)" << endl;
    }

    delete[] obs;
    delete[] value;
    delete[] piA;
    delete[] piB;
    delete[] wdl;
    delete[] movesLeft;
}


void UCI::loop() {
    string token, cmd;

    do {
        if (!getline(cin, cmd)) // Block here waiting for input or EOF
            cmd = "quit";

        istringstream is(cmd);

        token.clear(); // Avoid a stale if getline() returns empty or blank line
        is >> skipws >> token;

        try {
            if (token == "uci")             send_uci_response();
            else if (token == "isready")  { cout << "livenodes " << Node::live_count() << endl; cout << "readyok" << endl; }
            else if (token == "go")         go(is);
            else if (token == "ponderhit")  ponderhit();
            else if (token == "setoption")  setoption(is);
            else if (token == "position")   position(is);
            else if (token == "ucinewgame") new_game();
            else if (token == "stop")       stop();
            else if (token == "policy")     policy();
        } catch (const std::exception& error) {
            stop();
            cerr << "UCI command '" << token << "' failed: " << error.what() << endl;
            cout << "info string " << token << " failed: " << error.what() << endl;
        } catch (...) {
            stop();
            cerr << "UCI command '" << token << "' failed with an unknown exception" << endl;
            cout << "info string " << token << " failed: unknown exception" << endl;
        }

    } while (token != "quit"); // Command line args are one-shot
}
