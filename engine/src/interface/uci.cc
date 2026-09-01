#include "interface/uci.h"

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

    // Create the single-threaded Agent
    agent = std::make_unique<Agent>();
    return !engines.empty();
}


void UCI::stop() {
    if (agent) {
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
    if (agent) {
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
    if (!agent || engines.empty()) {
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
    
    // Launch the search thread
    mainSearchThread = new std::thread([this, enginePtrs, opts]() {
        try {
            const JointActionCandidate played = agent->run_search(
                board, enginePtrs, teamSide, teamHasTimeAdvantage, opts);
            if (!opts.isPonder) {
                // Nothing else runs between moves. Keep thinking from the
                // position our own move creates, so whichever reply the four
                // asynchronous players produce lands on a subtree that already
                // has work in it. Returns on the next position, go, or stop.
                agent->run_permanent_brain(
                    board, enginePtrs, teamSide, teamHasTimeAdvantage, played,
                    opts);
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

    while (ongoingSearch.load(std::memory_order_acquire) && !agent->is_running()) {
        std::this_thread::yield();
    }
}

void UCI::ponderhit() {
    if (agent && ongoingSearch.load(std::memory_order_acquire)) {
        agent->ponderhit();
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
        
        // Set hash size via Agent (which owns the transposition table)
        if (agent) {
            agent->setHashSize(sizeMB);
            std::cout << "info string Hash table set to " << sizeMB << " MB" << std::endl;
        }
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
    } else if (name == "RequireMoveOn") {
        g_requiredMoveBoard = parseRequiredMoveBoard(value);
        std::cout << "info string RequireMoveOn set to "
                  << (g_requiredMoveBoard == REQUIRE_MOVE_BOARD_A ? "A"
                      : g_requiredMoveBoard == REQUIRE_MOVE_BOARD_B ? "B"
                      : "none")
                  << std::endl;
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
    cout << "option name RequireMoveOn type combo default none var none var A var B" << endl;
    cout << "info string CUDA engines " << engines.size()
         << " search workers " << engines.size() * SearchParams::NUM_SEARCH_THREADS
         << " (" << SearchParams::NUM_SEARCH_THREADS << " per engine)" << endl;
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
