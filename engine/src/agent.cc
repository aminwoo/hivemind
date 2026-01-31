#include "agent.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "joint_action.h"
#include "search_params.h"
#include "utils.h"

using namespace std;

/**
 * @brief Format UCI score string based on node type.
 * 
 * Returns "score mate N" for proven wins/losses, "score cp X" otherwise.
 * Mate distance is computed from endInPly (ply to terminal).
 * Positive mate = we win in N moves, negative mate = we lose in N moves.
 * 
 * @param node The node to format score for
 * @param C Conversion constant for Q to centipawns
 * @param k Tan scaling constant
 * @return Formatted score string (e.g., "score cp 150" or "score mate 5")
 */
/**
 * @brief Format UCI score string based on node type and Q value.
 * 
 * Returns "score mate N" for proven wins/losses, "score cp X" otherwise.
 * Mate distance is computed from endInPly (ply to terminal).
 * Positive mate = we win in N moves, negative mate = we lose in N moves.
 * 
 * @param node The node to check for solved state
 * @param qFromParent The Q-value from the root's perspective (positive = good for root)
 * @param isChildNode True if this is a child node (opponent's perspective), false for root node
 * @param C Conversion constant for Q to centipawns
 * @param k Tan scaling constant
 * @return Formatted score string (e.g., "score cp 150" or "score mate 5")
 */
static string format_uci_score(const Node* node, float qFromParent, bool isChildNode = true, 
                               float C = 180.0f, float k = 1.56f) {
    if (!node) return "score cp 0";
    
    NodeType nodeType = node->get_node_type();
    int endInPly = node->get_end_in_ply();
    
    if (nodeType == NodeType::WIN) {
        int mateInMoves = (endInPly + 1) / 2;
        if (isChildNode) {
            // Child is a WIN for the child (opponent) = LOSS for us (we're mated)
            return "score mate -" + to_string(max(1, mateInMoves));
        } else {
            // Root is a WIN for us = we win
            return "score mate " + to_string(max(1, mateInMoves));
        }
    } else if (nodeType == NodeType::LOSS) {
        int mateInMoves = (endInPly + 1) / 2;
        if (isChildNode) {
            // Child is a LOSS for the child (opponent) = WIN for us (we mate them)
            return "score mate " + to_string(max(1, mateInMoves));
        } else {
            // Root is a LOSS for us = we're mated
            return "score mate -" + to_string(max(1, mateInMoves));
        }
    } else {
        // Not solved, use centipawn score from Q value (already from root's perspective)
        int cpScore = static_cast<int>(C * std::tan(k * qFromParent));
        return "score cp " + to_string(cpScore);
    }
}

/**
 * @brief Check if we should exit search early due to solved position or winning evaluation.
 * 
 * Returns true if:
 * - Root node is proven WIN (we have forced mate)
 * - Best child is proven LOSS (opponent loses = we have forced mate via that move)
 * - Best Q-value exceeds winning threshold and we have enough nodes searched
 * 
 * @param rootNode The root node of the search tree
 * @param bestChildIdx Index of the best child (by visits)
 * @param bestQ Q-value of the best child (from root's perspective)
 * @param nodesSearched Number of nodes searched so far
 * @param verbose If true, print info string when exiting early
 * @return True if search should exit early
 */
static bool should_exit_early_winning(const std::shared_ptr<Node>& rootNode, int bestChildIdx, 
                                       float bestQ, int nodesSearched, bool verbose) {
    if (!SearchParams::ENABLE_MATE_EARLY_EXIT) {
        return false;
    }
    
    if (!rootNode || !rootNode->is_expanded()) {
        return false;
    }
    
    // Check if root is proven WIN (we have forced mate)
    if (rootNode->get_node_type() == NodeType::WIN) {
        if (verbose) {
            cout << "info string Early exit: root position is proven WIN (forced mate)" << endl;
        }
        return true;
    }
    
    // Check if best child is proven LOSS (opponent loses = we win via that move)
    auto children = rootNode->get_children();
    if (bestChildIdx >= 0 && static_cast<size_t>(bestChildIdx) < children.size()) {
        Node* bestChild = children[bestChildIdx].get();
        if (bestChild && bestChild->get_node_type() == NodeType::LOSS) {
            if (verbose) {
                int mateInPly = bestChild->get_end_in_ply();
                int mateInMoves = (mateInPly + 1) / 2;
                cout << "info string Early exit: forced mate in " << mateInMoves << " found" << endl;
            }
            return true;
        }
    }
    
    // Check for overwhelmingly winning Q-value (only if enough nodes searched)
    if (nodesSearched >= SearchParams::MIN_NODES_FOR_Q_EXIT && 
        bestQ >= SearchParams::WINNING_Q_THRESHOLD) {
        if (verbose) {
            cout << "info string Early exit: position is completely winning (Q=" 
                 << std::fixed << std::setprecision(3) << bestQ << ")" << endl;
        }
        return true;
    }
    
    return false;
}

Agent::Agent(int numThreadsParam) : running(false), numThreads(0) {
    // Use specified thread count, or fall back to search params default
    numThreads = (numThreadsParam > 0) ? numThreadsParam : SearchParams::NUM_SEARCH_THREADS;
    
    // Create the transposition table for MCGS (if enabled)
    if (SearchParams::ENABLE_MCGS) {
        transpositionTable = std::make_unique<TranspositionTable>();
        transpositionTable->reserve(SearchParams::TT_INITIAL_CAPACITY);
    }
    
    // Start garbage collection thread for async tree cleanup
    gcThread_.start();
    
    // Create multiple search threads
    for (int i = 0; i < numThreads; i++) {
        searchThreads.push_back(new SearchThread());
    }
}

Agent::~Agent() {
    // Stop GC thread first
    gcThread_.stop();
    
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

    // Determine effective move time
    int moveTimeMs = options.moveTimeMs;
    size_t targetNodes = options.targetNodes;

    // Compute position hash for tree reuse
    uint64_t positionHash = board.hash_key(teamHasTimeAdvantage);
    
    // Try to reuse tree from previous search (if enabled)
    std::shared_ptr<Node> reusedRoot = nullptr;
    if (SearchParams::ENABLE_TREE_REUSE) {
        reusedRoot = try_reuse_tree(positionHash);
    }
    
    if (reusedRoot) {
        // Reuse the existing subtree
        rootNode = reusedRoot;
        rootNode->set_hash(positionHash);
        
        if (options.verbose) {
            cout << "info string Tree reuse: " << rootNode->get_visits() 
                 << " visits recovered" << endl;
        }
    } else {
        // Create new root node
        rootNode = make_shared<Node>(teamSide, positionHash);
    }
    
    SearchInfo searchInfo(chrono::steady_clock::now(), moveTimeMs);
    
    // MCGS: Clear and set up transposition table for new search (if enabled)
    if (SearchParams::ENABLE_MCGS && transpositionTable) {
        transpositionTable->clear();
        transpositionTable->insertOrGet(board.hash_key(teamHasTimeAdvantage), rootNode);
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
        // Workers check running flag and effective move time for time extension support
        if (moveTimeMs > 0) {
            workers.emplace_back([this, &board, engine, st, teamHasTimeAdvantage, &searchInfo]() {
                Board localBoard(board);
                while (running && searchInfo.elapsed() < searchInfo.get_effective_move_time()) {
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
    
    // Periodic info output during search (UCI verbose mode only)
    // Also handles early stopping and time extension
    constexpr int MIN_INFO_INTERVAL_MS = 100;
    if (options.verbose && moveTimeMs > 0) {
        searchInfo.set_in_game(true);
        constexpr float C = 180.0f;
        constexpr float k = 1.56f;
        int lastReportedDepth = 0;
        float lastCheckEval = 0.0f;
        bool evalInitialized = false;
        
        while (running && searchInfo.elapsed() < searchInfo.get_effective_move_time()) {
            // Sleep for remaining time or MIN_INFO_INTERVAL_MS, whichever is smaller
            double remainingMs = searchInfo.get_effective_move_time() - searchInfo.elapsed();
            int sleepMs = std::min(MIN_INFO_INTERVAL_MS, std::max(1, static_cast<int>(remainingMs)));
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
            
            if (!running || searchInfo.elapsed() >= searchInfo.get_effective_move_time()) break;
            
            // Update NPS tracking
            searchInfo.update_nps();
            
            int depth = searchInfo.get_max_depth();
            double elapsedMs = searchInfo.elapsed();
            int nodes = searchInfo.get_nodes_searched();
            int nps = (elapsedMs > 0) ? static_cast<int>((nodes * 1000.0) / elapsedMs) : 0;
            size_t tbhits = (SearchParams::ENABLE_MCGS && transpositionTable) ? transpositionTable->getHits() : 0;
            int hashfull = (SearchParams::ENABLE_MCGS && transpositionTable) ? transpositionTable->getFullness() : 0;
            
            if (rootNode && rootNode->is_expanded()) {
                auto childVisits = rootNode->get_child_visits();
                auto children = rootNode->get_children();
                size_t numChildren = min(childVisits.size(), children.size());
                
                if (numChildren > 0) {
                    // Find first and second max visit counts
                    int firstMax = 0, secondMax = 0;
                    int firstIdx = 0, secondIdx = -1;
                    for (size_t i = 0; i < numChildren; ++i) {
                        if (childVisits[i] > firstMax) {
                            secondMax = firstMax;
                            secondIdx = firstIdx;
                            firstMax = childVisits[i];
                            firstIdx = static_cast<int>(i);
                        } else if (childVisits[i] > secondMax) {
                            secondMax = childVisits[i];
                            secondIdx = static_cast<int>(i);
                        }
                    }
                    
                    float bestQ = rootNode->get_child_q(firstIdx);
                    float secondQ = (secondIdx >= 0) ? rootNode->get_child_q(secondIdx) : -1.0f;
                    
                    // Initialize eval tracking
                    if (!evalInitialized) {
                        lastCheckEval = bestQ;
                        searchInfo.set_last_eval(bestQ);
                        evalInitialized = true;
                    }
                    
                    // Early exit for solved/winning positions
                    if (should_exit_early_winning(rootNode, firstIdx, bestQ, nodes, true)) {
                        running = false;
                        break;
                    }
                    
                    // Early stopping check (visit-based)
                    if (SearchParams::ENABLE_EARLY_STOPPING && searchInfo.get_nps() > 0) {
                        double remaining = searchInfo.get_effective_move_time() - elapsedMs;
                        float projectedVisits = static_cast<float>(secondMax) + 
                                               static_cast<float>(remaining * searchInfo.get_nps() / 1000.0);
                        
                        // Stop if second-best can't catch up AND best move has better Q
                        if (projectedVisits < firstMax * SearchParams::EARLY_STOP_FACTOR &&
                            bestQ >= secondQ) {
                            cout << "info string Early stopping: saved " 
                                 << static_cast<int>(remaining) << "ms" << endl;
                            running = false;
                            break;
                        }
                    }
                    
                    // Time extension check - extend if eval is falling
                    if (SearchParams::ENABLE_TIME_EXTENSION && evalInitialized) {
                        float evalDrop = lastCheckEval - bestQ;
                        if (evalDrop > SearchParams::TIME_EXTENSION_THRESHOLD) {
                            if (searchInfo.try_extend_time(SearchParams::TIME_EXTENSION_FACTOR, 
                                                          SearchParams::MAX_TIME_EXTENSIONS)) {
                                cout << "info string Extending search time (eval dropped by " 
                                     << static_cast<int>(evalDrop * 100) << " cp)" << endl;
                            }
                        }
                        lastCheckEval = bestQ;
                    }
                    
                    // Only output when depth increases
                    if (depth > lastReportedDepth) {
                        lastReportedDepth = depth;
                        
                        // Create sorted indices by visit count (descending)
                        vector<size_t> sortedIndices(numChildren);
                        for (size_t i = 0; i < numChildren; ++i) sortedIndices[i] = i;
                        sort(sortedIndices.begin(), sortedIndices.end(), [&](size_t a, size_t b) {
                            return childVisits[a] > childVisits[b];
                        });
                        
                        // Output up to multiPV lines (only first PV during search to reduce overhead)
                        int numPVs = 1;  // Only output best line during search
                        for (int pvIdx = 0; pvIdx < numPVs; ++pvIdx) {
                            size_t childIdx = sortedIndices[pvIdx];
                            string pv = extract_pv_from_child(board, static_cast<int>(childIdx), 20);
                            float childQ = rootNode->get_child_q(static_cast<int>(childIdx));
                            string scoreStr = format_uci_score(children[childIdx].get(), childQ, true, C, k);
                            
                            cout << "info depth " << depth 
                                 << " " << scoreStr
                                 << " nodes " << nodes 
                                 << " nps " << nps
                                 << " hashfull " << hashfull
                                 << " tbhits " << tbhits
                                 << " time " << static_cast<int>(elapsedMs);
                            
                            if (!pv.empty()) {
                                cout << " pv " << pv;
                            }
                            cout << endl;
                        }
                    }
                }
            }
        }
    } else if (moveTimeMs > 0) {
        // Non-verbose mode: still check for early stopping
        searchInfo.set_in_game(true);
        float lastCheckEval = 0.0f;
        bool evalInitialized = false;
        
        while (running && searchInfo.elapsed() < searchInfo.get_effective_move_time()) {
            double remainingMs = searchInfo.get_effective_move_time() - searchInfo.elapsed();
            int sleepMs = std::min(100, std::max(1, static_cast<int>(remainingMs)));
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
            
            if (!running || searchInfo.elapsed() >= searchInfo.get_effective_move_time()) break;
            
            // Update NPS
            searchInfo.update_nps();
            
            if (rootNode && rootNode->is_expanded()) {
                auto childVisits = rootNode->get_child_visits();
                auto children = rootNode->get_children();
                size_t numChildren = min(childVisits.size(), children.size());
                
                if (numChildren > 0) {
                    // Find first and second max
                    int firstMax = 0, secondMax = 0;
                    int firstIdx = 0, secondIdx = -1;
                    for (size_t i = 0; i < numChildren; ++i) {
                        if (childVisits[i] > firstMax) {
                            secondMax = firstMax;
                            secondIdx = firstIdx;
                            firstMax = childVisits[i];
                            firstIdx = static_cast<int>(i);
                        } else if (childVisits[i] > secondMax) {
                            secondMax = childVisits[i];
                            secondIdx = static_cast<int>(i);
                        }
                    }
                    
                    float bestQ = rootNode->get_child_q(firstIdx);
                    float secondQ = (secondIdx >= 0) ? rootNode->get_child_q(secondIdx) : -1.0f;
                    
                    if (!evalInitialized) {
                        lastCheckEval = bestQ;
                        evalInitialized = true;
                    }
                    
                    int nodes = searchInfo.get_nodes_searched();
                    
                    // Early exit for solved/winning positions
                    if (should_exit_early_winning(rootNode, firstIdx, bestQ, nodes, false)) {
                        running = false;
                        break;
                    }
                    
                    // Early stopping (visit-based)
                    if (SearchParams::ENABLE_EARLY_STOPPING && searchInfo.get_nps() > 0) {
                        double remaining = searchInfo.get_effective_move_time() - searchInfo.elapsed();
                        float projectedVisits = static_cast<float>(secondMax) + 
                                               static_cast<float>(remaining * searchInfo.get_nps() / 1000.0);
                        
                        if (projectedVisits < firstMax * SearchParams::EARLY_STOP_FACTOR &&
                            bestQ >= secondQ) {
                            running = false;
                            break;
                        }
                    }
                    
                    // Time extension
                    if (SearchParams::ENABLE_TIME_EXTENSION && evalInitialized) {
                        float evalDrop = lastCheckEval - bestQ;
                        if (evalDrop > SearchParams::TIME_EXTENSION_THRESHOLD) {
                            searchInfo.try_extend_time(SearchParams::TIME_EXTENSION_FACTOR, 
                                                       SearchParams::MAX_TIME_EXTENSIONS);
                        }
                        lastCheckEval = bestQ;
                    }
                }
            }
        }
    } else {
        // Node-based search: wait for workers to reach target nodes
        // Workers will stop themselves when they've done enough iterations
        // Just need to wait and periodically check if all workers are done
        while (running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (static_cast<size_t>(searchInfo.get_nodes_searched()) >= targetNodes) {
                break;
            }
        }
    }
    
    // Signal workers to stop (in case they're still running)
    running = false;
    
    // Wait for all threads to complete
    for (auto& worker : workers) {
        worker.join();
    }

    // Extract best joint action by selecting the most visited child
    if (rootNode && rootNode->is_expanded()) {
        auto visits = rootNode->get_child_visits();
        auto children = rootNode->get_children();
        if (!visits.empty() && !children.empty()) {
            size_t numChildren = min(visits.size(), children.size());
            
            // Use Q-value weighted selection (with veto and weighting)
            int bestIdx = rootNode->get_best_move_idx_with_q_weight();
            
            // Fallback to most-visited if Q-value selection failed
            if (bestIdx < 0) {
                int maxVisits = 0;
                for (size_t i = 0; i < numChildren; ++i) {
                    if (visits[i] > maxVisits) {
                        maxVisits = visits[i];
                        bestIdx = static_cast<int>(i);
                    }
                }
            }
            
            size_t numGenerated = rootNode->get_num_generated();
            if (static_cast<size_t>(bestIdx) >= numGenerated) {
                cerr << "ERROR: bestIdx (" << bestIdx << ") >= numGenerated (" << numGenerated << ")" << endl;
                bestIdx = 0;
            }
            
            result = rootNode->get_joint_action(bestIdx);
        }
    }
    
    // Store next-root candidates for tree reuse
    if (SearchParams::ENABLE_TREE_REUSE) {
        store_next_root_candidates();
        lastSearchHash_ = board.hash_key(teamHasTimeAdvantage);
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
        
        // Multi-PV output: sort children by visits and output top N lines
        if (rootNode && rootNode->is_expanded()) {
            auto childVisits = rootNode->get_child_visits();
            auto children = rootNode->get_children();
            size_t numChildren = min(childVisits.size(), children.size());
            
            // Create sorted indices by visit count (descending)
            vector<size_t> sortedIndices(numChildren);
            for (size_t i = 0; i < numChildren; ++i) sortedIndices[i] = i;
            sort(sortedIndices.begin(), sortedIndices.end(), [&](size_t a, size_t b) {
                return childVisits[a] > childVisits[b];
            });
            
            // Output up to multiPV lines
            int numPVs = min(options.multiPV, static_cast<int>(numChildren));
            for (int pvIdx = 0; pvIdx < numPVs; ++pvIdx) {
                size_t childIdx = sortedIndices[pvIdx];
                string pv = extract_pv_from_child(board, static_cast<int>(childIdx), 20);
                float childQ = rootNode->get_child_q(static_cast<int>(childIdx));
                string scoreStr = format_uci_score(children[childIdx].get(), childQ, true, C, k);
                
                cout << "info depth " << depth 
                     << " multipv " << (pvIdx + 1)
                     << " " << scoreStr
                     << " nodes " << nodes 
                     << " nps " << nps
                     << " hashfull " << hashfull
                     << " tbhits " << tbhits
                     << " time " << static_cast<int>(elapsedMs);
                
                if (!pv.empty()) {
                    cout << " pv " << pv;
                }
                cout << endl;
            }
        } else {
            // Fallback: single PV line with root Q
            string pv = extract_pv(board, 20);
            // Use root's own Q value (which is from root's perspective)
            float rootQ = rootNode ? rootNode->Q() : 0.0f;
            string scoreStr = rootNode ? format_uci_score(rootNode.get(), rootQ, false, C, k) 
                                       : "score cp 0";
            
            cout << "info depth " << depth 
                 << " " << scoreStr
                 << " nodes " << nodes 
                 << " nps " << nps
                 << " hashfull " << hashfull
                 << " tbhits " << tbhits
                 << " time " << static_cast<int>(elapsedMs);
            
            if (!pv.empty()) {
                cout << " pv " << pv;
            }
            cout << endl;
        }

        string bestMoveStr = extract_best_move(board);
        cout << "bestmove " << bestMoveStr << endl;
    }
    
    return result;
}

/**
 * @brief Extracts the best move from the root node by selecting the most visited child.
 */
string Agent::extract_best_move(Board& board) {
    if (!rootNode || !rootNode->is_expanded()) {
        return "(none)";
    }

    auto childVisits = rootNode->get_child_visits();
    if (childVisits.empty()) {
        return "(none)";
    }

    // Find child with the most visits
    int bestIdx = 0;
    int maxVisits = 0;
    for (size_t i = 0; i < childVisits.size(); ++i) {
        if (childVisits[i] > maxVisits) {
            maxVisits = childVisits[i];
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
        
        // Sanity check: children and childVisits should have the same size
        if (children.size() != childVisits.size()) {
            cerr << "WARNING in extract_pv: children.size()=" << children.size()
                 << " != childVisits.size()=" << childVisits.size() 
                 << " at depth " << depth << endl;
        }
        
        // If debug log level, print all candidate moves at this PV node
        if (g_logLevel == LOG_DEBUG) {
            cout << "PV depth " << depth << " candidates:" << endl;
            for (size_t i = 0; i < children.size() && i < childVisits.size(); ++i) {
                JointActionCandidate candAction = currentNode->get_joint_action(static_cast<int>(i));
                string candMoveA = (candAction.moveA == Stockfish::MOVE_NONE) ? "pass" : tempBoard.uci_move(BOARD_A, candAction.moveA);
                string candMoveB = (candAction.moveB == Stockfish::MOVE_NONE) ? "pass" : tempBoard.uci_move(BOARD_B, candAction.moveB);
                float candQ = currentNode->get_child_q(static_cast<int>(i));
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
        
        // Verify the action is valid - if both moves are MOVE_NONE, it should be intentional
        size_t genCount = currentNode->get_num_generated();
        if (static_cast<size_t>(bestIdx) >= genCount) {
            cerr << "WARNING in extract_pv: bestIdx=" << bestIdx 
                 << " >= generatedCount=" << genCount 
                 << " at depth " << depth << endl;
        }
        
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

/**
 * @brief Extracts PV line starting from a specific child index.
 * Used for Multi-PV output to show principal variations for non-best moves.
 * @param board The current board position
 * @param childIdx The child index to start the PV from
 * @param maxDepth Maximum number of moves to extract
 * @return Space-separated sequence of joint moves in format "(moveA,moveB) (moveA,moveB) ..."
 */
string Agent::extract_pv_from_child(Board& board, int childIdx, int maxDepth) {
    if (!rootNode || !rootNode->is_expanded()) {
        return "";
    }
    
    auto children = rootNode->get_children();
    if (childIdx < 0 || static_cast<size_t>(childIdx) >= children.size()) {
        return "";
    }
    
    Board tempBoard = board;
    string pv;
    
    // Get the first move from the specified child
    JointActionCandidate action = rootNode->get_joint_action(childIdx);
    string moveA = (action.moveA == Stockfish::MOVE_NONE) 
                    ? "pass" : tempBoard.uci_move(BOARD_A, action.moveA);
    string moveB = (action.moveB == Stockfish::MOVE_NONE) 
                    ? "pass" : tempBoard.uci_move(BOARD_B, action.moveB);
    pv = "(" + moveA + "," + moveB + ")";
    
    // Apply moves to temp board
    tempBoard.make_moves(action.moveA, action.moveB);
    
    // Continue extracting PV from this child
    Node* currentNode = children[childIdx].get();
    
    for (int depth = 1; depth < maxDepth; depth++) {
        if (!currentNode || !currentNode->is_expanded()) {
            break;
        }
        
        auto nodeChildren = currentNode->get_children();
        auto childVisits = currentNode->get_child_visits();
        if (nodeChildren.empty() || childVisits.empty()) {
            break;
        }
        
        // Find child with most visits
        int bestIdx = 0;
        int maxVisits = 0;
        for (size_t i = 0; i < nodeChildren.size() && i < childVisits.size(); i++) {
            if (childVisits[i] > maxVisits) {
                maxVisits = childVisits[i];
                bestIdx = static_cast<int>(i);
            }
        }
        
        // Get the joint action for this move
        JointActionCandidate nextAction = currentNode->get_joint_action(bestIdx);
        
        // Format move string
        string nextMoveA = (nextAction.moveA == Stockfish::MOVE_NONE) 
                            ? "pass" : tempBoard.uci_move(BOARD_A, nextAction.moveA);
        string nextMoveB = (nextAction.moveB == Stockfish::MOVE_NONE) 
                            ? "pass" : tempBoard.uci_move(BOARD_B, nextAction.moveB);
        
        pv += " (" + nextMoveA + "," + nextMoveB + ")";
        
        // Apply moves to temp board for next iteration
        tempBoard.make_moves(nextAction.moveA, nextAction.moveB);
        
        // Move to best child
        currentNode = nodeChildren[bestIdx].get();
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

/**
 * @brief Try to reuse the search tree from a previous search.
 * 
 * Implements CrazyAra-style tree reuse by checking if the current position
 * matches either our expected move (ownNextRoot_) or opponent's expected
 * response (opponentsNextRoot_).
 */
std::shared_ptr<Node> Agent::try_reuse_tree(uint64_t positionHash) {
    // Check if ownNextRoot_ matches (we made our expected move)
    if (ownNextRoot_ && ownNextRoot_->get_hash() == positionHash) {
        auto reused = ownNextRoot_;
        
        // Queue old root for garbage collection (if different from reused)
        if (rootNode && rootNode != reused) {
            gcThread_.enqueue(rootNode);
        }
        
        // Clear both candidates since we're reusing one
        ownNextRoot_.reset();
        opponentsNextRoot_.reset();
        
        return reused;
    }
    
    // Check if opponentsNextRoot_ matches (opponent made expected response)
    if (opponentsNextRoot_ && opponentsNextRoot_->get_hash() == positionHash) {
        auto reused = opponentsNextRoot_;
        
        // Queue old root for garbage collection
        if (rootNode && rootNode != reused) {
            gcThread_.enqueue(rootNode);
        }
        
        // Clear both candidates
        ownNextRoot_.reset();
        opponentsNextRoot_.reset();
        
        return reused;
    }
    
    // No reuse possible - queue old tree for GC
    if (rootNode) {
        gcThread_.enqueue(rootNode);
    }
    
    // Clear candidates since they're now invalid
    ownNextRoot_.reset();
    opponentsNextRoot_.reset();
    
    return nullptr;
}

/**
 * @brief Store next-root candidates for tree reuse.
 * 
 * After search completes, store references to likely next positions:
 * - ownNextRoot_: Best child (our expected move)
 * - opponentsNextRoot_: Best grandchild (opponent's expected response)
 */
void Agent::store_next_root_candidates() {
    if (!rootNode || !rootNode->is_expanded()) {
        ownNextRoot_.reset();
        opponentsNextRoot_.reset();
        return;
    }
    
    auto children = rootNode->get_children();
    auto visits = rootNode->get_child_visits();
    
    if (children.empty() || visits.empty()) {
        ownNextRoot_.reset();
        opponentsNextRoot_.reset();
        return;
    }
    
    // Find best child (most visited, with Q-value consideration)
    int bestIdx = rootNode->get_best_move_idx_with_q_weight();
    if (bestIdx < 0) {
        // Fallback to most-visited
        int maxVisits = 0;
        for (size_t i = 0; i < visits.size(); ++i) {
            if (visits[i] > maxVisits) {
                maxVisits = visits[i];
                bestIdx = static_cast<int>(i);
            }
        }
    }
    
    if (bestIdx >= 0 && static_cast<size_t>(bestIdx) < children.size()) {
        ownNextRoot_ = children[bestIdx];
        
        // Find best grandchild (opponent's expected response)
        if (ownNextRoot_ && ownNextRoot_->is_expanded()) {
            auto grandchildren = ownNextRoot_->get_children();
            auto grandVisits = ownNextRoot_->get_child_visits();
            
            if (!grandchildren.empty() && !grandVisits.empty()) {
                int bestGrandIdx = 0;
                int maxGrandVisits = 0;
                for (size_t i = 0; i < grandVisits.size(); ++i) {
                    if (grandVisits[i] > maxGrandVisits) {
                        maxGrandVisits = grandVisits[i];
                        bestGrandIdx = static_cast<int>(i);
                    }
                }
                
                if (static_cast<size_t>(bestGrandIdx) < grandchildren.size()) {
                    opponentsNextRoot_ = grandchildren[bestGrandIdx];
                }
            }
        }
    }
}
