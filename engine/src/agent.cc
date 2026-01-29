#include "agent.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "joint_action.h"
#include "search_params.h"
#include "utils.h"

using namespace std;

Agent::Agent(int numThreadsParam) : running(false), numThreads(0) {
    // Use specified thread count, or fall back to search params default
    numThreads = (numThreadsParam > 0) ? numThreadsParam : SearchParams::NUM_SEARCH_THREADS;
    
    // Create the transposition table for MCGS (if enabled)
    if (SearchParams::ENABLE_MCGS) {
        transpositionTable = std::make_unique<TranspositionTable>();
        transpositionTable->reserve(SearchParams::TT_INITIAL_CAPACITY);
    }
    
    // Create multiple search threads
    for (int i = 0; i < numThreads; i++) {
        searchThreads.push_back(new SearchThread());
    }
}

Agent::~Agent() {
    for (auto* st : searchThreads) {
        delete st;
    }
    searchThreads.clear();
}

/**
 * @brief Unified search function for both UCI and self-play modes.
 */
JointActionCandidate Agent::run_search(Board& board, const vector<Engine*>& engines, 
                                        Stockfish::Color teamSide, bool teamHasTimeAdvantage,
                                        const SearchOptions& options) {
    JointActionCandidate result;
    
    // Check for no legal moves
    if (board.legal_moves(teamSide, teamHasTimeAdvantage).empty() || 
        board.is_checkmate(~teamSide, !teamHasTimeAdvantage)) {
        if (options.verbose) {
            cout << "bestmove (none)" << endl;
        }
        return result;
    }

    // Check for mate in 1 on either board before starting search (UCI mode only)
    if (options.checkMateIn1) {
        auto legalMoves = board.legal_moves(teamSide, teamHasTimeAdvantage);
        for (const auto& [boardNum, move] : legalMoves) {
            if (boardNum == BOARD_A) {
                board.make_moves(move, Stockfish::MOVE_NONE);
            } else {
                board.make_moves(Stockfish::MOVE_NONE, move);
            }
            
            if (board.is_checkmate(~teamSide, !teamHasTimeAdvantage)) {
                if (boardNum == BOARD_A) {
                    board.unmake_moves(move, Stockfish::MOVE_NONE);
                } else {
                    board.unmake_moves(Stockfish::MOVE_NONE, move);
                }
                
                string moveStr = board.uci_move(boardNum, move);
                string moveA = (boardNum == BOARD_A) ? moveStr : "pass";
                string moveB = (boardNum == BOARD_B) ? moveStr : "pass";
                
                if (options.verbose) {
                    cout << "info depth 1 score mate 1 nodes 1 nps 0 time 0" << endl;
                    cout << "bestmove (" << moveA << "," << moveB << ")" << endl;
                }
                
                result.moveA = (boardNum == BOARD_A) ? move : Stockfish::MOVE_NONE;
                result.moveB = (boardNum == BOARD_B) ? move : Stockfish::MOVE_NONE;
                return result;
            }
            
            if (boardNum == BOARD_A) {
                board.unmake_moves(move, Stockfish::MOVE_NONE);
            } else {
                board.unmake_moves(Stockfish::MOVE_NONE, move);
            }
        }
    }

    // Determine effective move time
    int moveTimeMs = options.moveTimeMs;
    size_t targetNodes = options.targetNodes;

    // Create the root node and search info
    rootNode = make_shared<Node>(teamSide, board.hash_key(teamHasTimeAdvantage));
    SearchInfo searchInfo(chrono::steady_clock::now(), moveTimeMs);
    
    // MCGS: Clear and set up transposition table for new search (if enabled)
    if (SearchParams::ENABLE_MCGS && transpositionTable) {
        transpositionTable->clear();
        transpositionTable->insertOrGet(board.hash_key(teamHasTimeAdvantage), rootNode);
    }

    if (options.verbose && engines.size() < static_cast<size_t>(numThreads)) {
        cerr << "Warning: Not enough engines for all threads. Need " 
             << numThreads << ", have " << engines.size() << endl;
    }

    // Set up all search threads with shared root node, search info, and transposition table
    for (auto* st : searchThreads) {
        st->set_root_node(rootNode.get());
        st->set_search_info(&searchInfo);
        if (SearchParams::ENABLE_MCGS) {
            st->set_transposition_table(transpositionTable.get());
        }
    }
    
    running = true;

    // If Dirichlet noise is enabled, run one iteration first to expand root,
    // then apply noise before main search
    bool applyDirichlet = (options.dirichletEpsilon > 0.0f);
    if (applyDirichlet) {
        Engine* engine = engines[0];
        SearchThread* st = searchThreads[0];
        Board localBoard(board);
        st->run_iteration(localBoard, engine, teamHasTimeAdvantage);
        
        if (rootNode && rootNode->is_expanded()) {
            size_t numChildren = rootNode->get_num_children();
            if (numChildren > 0) {
                auto noise = generate_dirichlet_noise(numChildren, options.dirichletAlpha);
                rootNode->apply_dirichlet_noise(noise, options.dirichletEpsilon);
            }
        }
    }

    // Launch worker threads
    vector<thread> workers;
    for (int i = 0; i < numThreads; i++) {
        Engine* engine = engines[i % engines.size()];
        SearchThread* st = searchThreads[i];
        
        // Use time-based stopping if moveTimeMs > 0, otherwise use node-based
        if (moveTimeMs > 0) {
            workers.emplace_back([this, &board, engine, st, moveTimeMs, teamHasTimeAdvantage, &searchInfo]() {
                Board localBoard(board);
                while (running && searchInfo.elapsed() < moveTimeMs) {
                    st->run_iteration(localBoard, engine, teamHasTimeAdvantage);
                }
            });
        } else {
            workers.emplace_back([this, &board, engine, st, targetNodes, teamHasTimeAdvantage, &searchInfo]() {
                Board localBoard(board);
                while (running && static_cast<size_t>(searchInfo.get_nodes_searched()) < targetNodes) {
                    st->run_iteration(localBoard, engine, teamHasTimeAdvantage);
                }
            });
        }
    }
    
    // Wait for all threads to complete
    for (auto& worker : workers) {
        worker.join();
    }

    running = false;
    
    // Extract best joint action (greedy by visit count)
    if (rootNode && rootNode->is_expanded()) {
        auto visits = rootNode->get_child_visits();
        if (!visits.empty()) {
            size_t bestIdx = 0;
            int maxVisits = visits[0];
            float bestQ = -2.0f;
            auto children = rootNode->get_children();
            
            for (size_t i = 0; i < visits.size(); ++i) {
                float q = (i < children.size()) ? children[i]->Q() : -2.0f;
                if (visits[i] > maxVisits || (visits[i] == maxVisits && q > bestQ)) {
                    maxVisits = visits[i];
                    bestQ = q;
                    bestIdx = i;
                }
            }
            
            size_t numGenerated = rootNode->get_num_generated();
            if (bestIdx >= numGenerated) {
                cerr << "ERROR: bestIdx (" << bestIdx << ") >= numGenerated (" << numGenerated << ")" << endl;
                bestIdx = 0;
            }
            
            result = rootNode->get_joint_action(static_cast<int>(bestIdx));
        }
    }
    
    // Output UCI info if verbose
    if (options.verbose) {
        double elapsedMs = searchInfo.elapsed();
        int nodes = searchInfo.get_nodes_searched();
        int depth = searchInfo.get_max_depth();
        int nps = (elapsedMs > 0) ? static_cast<int>((nodes * 1000.0) / elapsedMs) : 0;
        size_t tbhits = (SearchParams::ENABLE_MCGS && transpositionTable) ? transpositionTable->getHits() : 0;
        int hashfull = (SearchParams::ENABLE_MCGS && transpositionTable) ? transpositionTable->getFullness() : 0;
        
        // Convert Q-value [-1, 1] to centipawns using Lc0 tangent formula
        constexpr float C = 180.0f;
        constexpr float k = 1.56f;
        float q = rootNode ? rootNode->Q() : 0.0f;
        int cpScore = static_cast<int>(C * std::tan(k * q));
        
        string pv = extract_pv(board, 20);
        
        cout << "info depth " << depth 
             << " score cp " << cpScore
             << " nodes " << nodes 
             << " nps " << nps
             << " hashfull " << hashfull
             << " tbhits " << tbhits
             << " time " << static_cast<int>(elapsedMs);
        
        if (!pv.empty()) {
            cout << " pv " << pv;
        }
        cout << endl;

        string bestMoveStr = extract_best_move(board);
        cout << "bestmove " << bestMoveStr << endl;
    }
    
    return result;
}

// Legacy wrappers for backwards compatibility
void Agent::run_search(Board& board, const vector<Engine*>& engines, int moveTime, Stockfish::Color teamSide, bool teamHasTimeAdvantage) {
    auto opts = SearchOptions::uci(moveTime);
    run_search(board, engines, teamSide, teamHasTimeAdvantage, opts);
}

JointActionCandidate Agent::run_search_silent(Board& board, const vector<Engine*>& engines, size_t targetNodes, int moveTimeMs, Stockfish::Color teamSide, bool teamHasTimeAdvantage, const RLSettings& settings, float temperature) {
    SearchOptions opts;
    opts.targetNodes = targetNodes;
    opts.moveTimeMs = moveTimeMs;
    opts.verbose = false;
    opts.checkMateIn1 = false;
    opts.dirichletAlpha = settings.dirichletAlpha;
    opts.dirichletEpsilon = settings.dirichletEpsilon;
    return run_search(board, engines, teamSide, teamHasTimeAdvantage, opts);
}

/**
 * @brief Extracts the best move from the root node based on visit counts and Q-values.
 */
string Agent::extract_best_move(Board& board) {
    if (!rootNode || !rootNode->is_expanded()) {
        return "(none)";
    }

    // Find best child by visit count (with Q-value tiebreaker)
    // Use parent's childVisits array which is correctly updated during backup
    auto childVisits = rootNode->get_child_visits();
    auto children = rootNode->get_children();
    if (children.empty() || childVisits.empty()) {
        return "(none)";
    }

    int bestIdx = 0;
    int maxVisits = 0;
    float bestQ = -2.0f;

    for (size_t i = 0; i < children.size() && i < childVisits.size(); i++) {
        int visits = childVisits[i];
        float q = children[i]->Q();
        
        if (visits > maxVisits || (visits == maxVisits && q > bestQ)) {
            maxVisits = visits;
            bestQ = q;
            bestIdx = static_cast<int>(i);
        }
    }

    JointActionCandidate action = rootNode->get_joint_action(bestIdx);
    string moveA = (action.moveA == Stockfish::MOVE_NONE) 
                    ? "pass" : board.uci_move(BOARD_A, action.moveA);
    string moveB = (action.moveB == Stockfish::MOVE_NONE) 
                    ? "pass" : board.uci_move(BOARD_B, action.moveB);
    return "(" + moveA + "," + moveB + ")";
}

/**
 * @brief Extracts the principal variation (PV) by following most-visited children.
 * @param board The current board position
 * @param maxDepth Maximum number of moves to extract
 * @return Space-separated sequence of joint moves in format "(moveA,moveB) (moveA,moveB) ..."
 */
string Agent::extract_pv(Board& board, int maxDepth) {
    if (!rootNode || !rootNode->is_expanded()) {
        return "";
    }
    
    Board tempBoard = board;  // Make a copy to simulate moves
    Node* currentNode = rootNode.get();
    string pv;
    
    for (int depth = 0; depth < maxDepth; depth++) {
        if (!currentNode || !currentNode->is_expanded()) {
            break;
        }
        
        auto children = currentNode->get_children();
        auto childVisits = currentNode->get_child_visits();
        if (children.empty() || childVisits.empty()) {
            break;
        }
        
        // If debug log level, print all candidate moves at this PV node
        if (g_logLevel == LOG_DEBUG) {
            cout << "PV depth " << depth << " candidates:" << endl;
            for (size_t i = 0; i < children.size() && i < childVisits.size(); ++i) {
                JointActionCandidate candAction = currentNode->get_joint_action(static_cast<int>(i));
                string candMoveA = (candAction.moveA == Stockfish::MOVE_NONE) ? "pass" : tempBoard.uci_move(BOARD_A, candAction.moveA);
                string candMoveB = (candAction.moveB == Stockfish::MOVE_NONE) ? "pass" : tempBoard.uci_move(BOARD_B, candAction.moveB);
                float candQ = children[i]->Q();
                int candVisitsCount = childVisits[i];
                cout << "  (" << candMoveA << ", " << candMoveB << ")"
                     << "  Q: " << std::fixed << std::setprecision(3) << candQ
                     << "  Visits: " << candVisitsCount << endl;
            }
        }
        
        // Find child with most visits (use parent's childVisits array)
        int bestIdx = 0;
        int maxVisits = 0;
        
        for (size_t i = 0; i < children.size() && i < childVisits.size(); i++) {
            int visits = childVisits[i];
            if (visits > maxVisits) {
                maxVisits = visits;
                bestIdx = static_cast<int>(i);
            }
        }
        
        // Get the joint action for this move
        JointActionCandidate action = currentNode->get_joint_action(bestIdx);
        
        // Format move string
        string moveA = (action.moveA == Stockfish::MOVE_NONE) 
                        ? "pass" : tempBoard.uci_move(BOARD_A, action.moveA);
        string moveB = (action.moveB == Stockfish::MOVE_NONE) 
                        ? "pass" : tempBoard.uci_move(BOARD_B, action.moveB);
        
        if (!pv.empty()) {
            pv += " ";
        }
        pv += "(" + moveA + "," + moveB + ")";
        
        // Apply moves to temp board for next iteration
        tempBoard.make_moves(action.moveA, action.moveB);
        
        // Move to best child
        currentNode = children[bestIdx].get();
    }
    
    return pv;
}

void Agent::set_is_running(bool value) {
    running = value;
}

bool Agent::is_running() {
    return running;
}

void Agent::setHashSize(size_t sizeMB) {
    // Clamp to valid range (1 MB to 32 TB)
    sizeMB = std::max(static_cast<size_t>(1), std::min(sizeMB, static_cast<size_t>(33554432)));
    
    // Convert MB to approximate entry count
    // Each TT entry is roughly 64 bytes (hash key + shared_ptr + unordered_map overhead)
    constexpr size_t BYTES_PER_ENTRY = 64;
    size_t maxEntries = (sizeMB * 1024 * 1024) / BYTES_PER_ENTRY;
    
    if (!transpositionTable && SearchParams::ENABLE_MCGS) {
        transpositionTable = std::make_unique<TranspositionTable>();
    }
    
    if (transpositionTable) {
        transpositionTable->setMaxCapacity(maxEntries);
        transpositionTable->reserve(maxEntries);
        transpositionTable->clear();
    }
}
