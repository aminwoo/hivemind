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

using namespace std;

// Thread-local random generator for Dirichlet noise
static thread_local std::mt19937 rng(std::random_device{}());

/**
 * @brief Generate Dirichlet noise vector.
 * @param length Number of elements
 * @param alpha Concentration parameter
 * @return Normalized Dirichlet sample
 */
static vector<float> generate_dirichlet_noise(size_t length, float alpha) {
    vector<float> noise(length);
    float sum = 0.0f;
    
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    for (size_t i = 0; i < length; ++i) {
        noise[i] = gamma(rng);
        sum += noise[i];
    }
    
    // Normalize to sum to 1
    if (sum > 0.0f) {
        for (size_t i = 0; i < length; ++i) {
            noise[i] /= sum;
        }
    }
    
    return noise;
}

/**
 * @brief Sample an index based on visit counts with temperature.
 * @param visits Vector of visit counts
 * @param temperature Temperature for sampling (0 = greedy, >0 = stochastic)
 * @return Sampled index
 */
static size_t sample_with_temperature(const vector<int>& visits, float temperature) {
    if (visits.empty()) return 0;
    
    // Greedy selection for temperature near 0
    if (temperature < 0.01f) {
        size_t bestIdx = 0;
        int maxVisits = visits[0];
        for (size_t i = 1; i < visits.size(); ++i) {
            if (visits[i] > maxVisits) {
                maxVisits = visits[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }
    
    // Temperature-based sampling: P(i) = visits[i]^(1/T) / sum(visits^(1/T))
    vector<double> probs(visits.size());
    double sum = 0.0;
    double invTemp = 1.0 / temperature;
    
    for (size_t i = 0; i < visits.size(); ++i) {
        probs[i] = pow(static_cast<double>(visits[i]), invTemp);
        sum += probs[i];
    }
    
    if (sum <= 0.0) {
        return 0;
    }
    
    // Normalize
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum;
    }
    
    // Sample
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng);
    double cumulative = 0.0;
    
    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative += probs[i];
        if (r <= cumulative) {
            return i;
        }
    }
    
    return visits.size() - 1;
}

Agent::Agent() : running(false) {
    // Create the transposition table for MCGS (if enabled)
    if (SearchParams::ENABLE_MCGS) {
        transpositionTable = std::make_unique<TranspositionTable>();
        transpositionTable->reserve(SearchParams::TT_INITIAL_CAPACITY);
    }
    
    // Create multiple search threads
    for (int i = 0; i < SearchParams::NUM_SEARCH_THREADS; i++) {
        searchThreads.push_back(new SearchThread());
    }
}

Agent::~Agent() {
    for (auto* st : searchThreads) {
        delete st;
    }
    searchThreads.clear();
}

void Agent::run_search(Board& board, const vector<Engine*>& engines, int moveTime, Stockfish::Color teamSide, bool teamHasTimeAdvantage) {
    if (board.legal_moves(teamSide, teamHasTimeAdvantage).empty()) {
        cout << "bestmove (none)" << endl;
        return;
    }

    // Check for mate in 1 on either board before starting search
    auto legalMoves = board.legal_moves(teamSide, teamHasTimeAdvantage);
    for (const auto& [boardNum, move] : legalMoves) {
        if (boardNum == 0) {
            board.make_moves(move, Stockfish::MOVE_NONE);
        } else {
            board.make_moves(Stockfish::MOVE_NONE, move);
        }
        
        // Check if opponent is checkmated - opponent's time advantage is inverse of ours
        if (board.is_checkmate(~teamSide, !teamHasTimeAdvantage)) {
            if (boardNum == 0) {
                board.unmake_moves(move, Stockfish::MOVE_NONE);
            } else {
                board.unmake_moves(Stockfish::MOVE_NONE, move);
            }
            
            string moveStr = board.uci_move(boardNum, move);
            string moveA = (boardNum == 0) ? moveStr : "pass";
            string moveB = (boardNum == 1) ? moveStr : "pass";
            
            cout << "info depth 1 score mate 1 nodes 1 nps 0 time 0" << endl;
            cout << "bestmove (" << moveA << "," << moveB << ")" << endl;
            return;
        }
        
        if (boardNum == 0) {
            board.unmake_moves(move, Stockfish::MOVE_NONE);
        } else {
            board.unmake_moves(Stockfish::MOVE_NONE, move);
        }
    }

    // Create the root node and search info
    rootNode = make_shared<Node>(teamSide, board.hash_key(teamHasTimeAdvantage));  // MCGS: store root hash
    SearchInfo searchInfo(chrono::steady_clock::now(), moveTime);
    
    // MCGS: Clear and set up transposition table for new search (if enabled)
    if (SearchParams::ENABLE_MCGS && transpositionTable) {
        transpositionTable->clear();
        transpositionTable->insertOrGet(board.hash_key(teamHasTimeAdvantage), rootNode);
    }

    if (engines.size() < static_cast<size_t>(SearchParams::NUM_SEARCH_THREADS)) {
        cerr << "Warning: Not enough engines for all threads. Need " 
             << SearchParams::NUM_SEARCH_THREADS << ", have " << engines.size() << endl;
    }

    // Set up all search threads with shared root node, search info, and transposition table
    for (auto* st : searchThreads) {
        st->set_root_node(rootNode.get());
        st->set_search_info(&searchInfo);
        if (SearchParams::ENABLE_MCGS) {
            st->set_transposition_table(transpositionTable.get());  // MCGS
        }
    }
    
    running = true;

    // Launch worker threads
    vector<thread> workers;
    for (int i = 0; i < SearchParams::NUM_SEARCH_THREADS; i++) {
        // Each thread gets its own engine (or shares if not enough)
        Engine* engine = engines[i % engines.size()];
        SearchThread* st = searchThreads[i];
        
        workers.emplace_back([this, &board, engine, st, moveTime, teamHasTimeAdvantage, &searchInfo]() {
            // Each thread needs its own copy of the board for traversal
            Board localBoard(board);
            
            while (running && searchInfo.elapsed() < moveTime) {
                st->run_iteration(localBoard, engine, teamHasTimeAdvantage);
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& worker : workers) {
        worker.join();
    }

    running = false;
    
    // Calculate search statistics
    double elapsedMs = searchInfo.elapsed();
    int nodes = searchInfo.get_nodes_searched();
    int depth = searchInfo.get_max_depth();
    int collisions = searchInfo.get_collisions();
    int nps = (elapsedMs > 0) ? static_cast<int>((nodes * 1000.0) / elapsedMs) : 0;
    size_t tbhits = (SearchParams::ENABLE_MCGS && transpositionTable) ? transpositionTable->getHits() : 0;
    int hashfull = (SearchParams::ENABLE_MCGS && transpositionTable) ? transpositionTable->getFullness() : 0;
    
    // Convert Q-value [-1, 1] to centipawns using Lc0 tangent formula
    // cp = C * tan(k * Q), where C=180 and k=1.56 (tuned for Bughouse)
    // Higher scaling reflects the increased value of material in Bughouse
    constexpr float C = 180.0f;  // Scaling constant
    constexpr float k = 1.56f;   // Curvature constant
    float q = rootNode->Q();
    int cpScore = static_cast<int>(C * std::tan(k * q));
    
    // Extract principal variation (PV) - sequence of most visited moves
    string pv = extract_pv(board, 30);  // Extract up to 20 moves
    
    // Output UCI info string
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

/**
 * @brief Runs a silent search for self-play (no UCI output).
 * Returns the best joint action found.
 * @param targetNodes Number of MCTS iterations to run.
 * @param settings RL settings for Dirichlet noise.
 * @param temperature Temperature for move selection.
 */
JointActionCandidate Agent::run_search_silent(Board& board, const vector<Engine*>& engines, size_t targetNodes, Stockfish::Color teamSide, bool teamHasTimeAdvantage, const RLSettings& settings, float temperature) {
    JointActionCandidate result;
    
    if (board.legal_moves(teamSide, teamHasTimeAdvantage).empty()) {
        return result;  // No legal moves
    }

    // Check for mate in 1 on either board before starting search
    auto legalMoves = board.legal_moves(teamSide, teamHasTimeAdvantage);
    for (const auto& [boardNum, move] : legalMoves) {
        if (boardNum == 0) {
            board.make_moves(move, Stockfish::MOVE_NONE);
        } else {
            board.make_moves(Stockfish::MOVE_NONE, move);
        }
        
        if (board.is_checkmate(~teamSide, !teamHasTimeAdvantage)) {
            if (boardNum == 0) {
                board.unmake_moves(move, Stockfish::MOVE_NONE);
                result.moveA = move;
                result.moveB = Stockfish::MOVE_NONE;
            } else {
                board.unmake_moves(Stockfish::MOVE_NONE, move);
                result.moveA = Stockfish::MOVE_NONE;
                result.moveB = move;
            }
            return result;
        }
        
        if (boardNum == 0) {
            board.unmake_moves(move, Stockfish::MOVE_NONE);
        } else {
            board.unmake_moves(Stockfish::MOVE_NONE, move);
        }
    }

    // Create the root node and search info
    rootNode = make_shared<Node>(teamSide, board.hash_key(teamHasTimeAdvantage));
    SearchInfo searchInfo(chrono::steady_clock::now(), 0);  // Time not used for node-based search
    
    if (SearchParams::ENABLE_MCGS && transpositionTable) {
        transpositionTable->clear();
        transpositionTable->insertOrGet(board.hash_key(teamHasTimeAdvantage), rootNode);
    }

    for (auto* st : searchThreads) {
        st->set_root_node(rootNode.get());
        st->set_search_info(&searchInfo);
        if (SearchParams::ENABLE_MCGS) {
            st->set_transposition_table(transpositionTable.get());
        }
    }
    
    running = true;

    // Launch worker threads
    vector<thread> workers;
    for (int i = 0; i < SearchParams::NUM_SEARCH_THREADS; i++) {
        Engine* engine = engines[i % engines.size()];
        SearchThread* st = searchThreads[i];
        
        workers.emplace_back([this, &board, engine, st, targetNodes, teamHasTimeAdvantage, &searchInfo]() {
            Board localBoard(board);
            while (running && static_cast<size_t>(searchInfo.get_nodes_searched()) < targetNodes) {
                st->run_iteration(localBoard, engine, teamHasTimeAdvantage);
            }
        });
    }
    
    for (auto& worker : workers) {
        worker.join();
    }

    running = false;
    
    // Apply Dirichlet noise to root node priors after expansion
    // (for self-play exploration)
    if (rootNode && rootNode->is_expanded() && settings.dirichletEpsilon > 0.0f) {
        size_t numChildren = rootNode->get_num_children();
        if (numChildren > 0) {
            auto noise = generate_dirichlet_noise(numChildren, settings.dirichletAlpha);
            rootNode->apply_dirichlet_noise(noise, settings.dirichletEpsilon);
        }
    }
    
    // Extract best joint action with temperature-based selection
    if (rootNode && rootNode->is_expanded()) {
        auto children = rootNode->get_children();
        if (!children.empty()) {
            // Collect visit counts
            vector<int> visits;
            visits.reserve(children.size());
            for (const auto& child : children) {
                visits.push_back(child->get_visits());
            }
            
            // Sample index with temperature
            size_t selectedIdx = sample_with_temperature(visits, temperature);
            result = rootNode->get_joint_action(static_cast<int>(selectedIdx));
        }
    }
    
    return result;
}

/**
 * @brief Extracts the best move from the root node based on visit counts and Q-values.
 */
string Agent::extract_best_move(Board& board) {
    if (!rootNode || !rootNode->is_expanded()) {
        return "(none)";
    }

    // Find best child by visit count (with Q-value tiebreaker)
    auto children = rootNode->get_children();
    if (children.empty()) {
        return "(none)";
    }

    int bestIdx = 0;
    int maxVisits = 0;
    float bestQ = -2.0f;

    for (size_t i = 0; i < children.size(); i++) {
        int visits = children[i]->get_visits();
        float q = children[i]->Q();
        
        if (visits > maxVisits || (visits == maxVisits && q > bestQ)) {
            maxVisits = visits;
            bestQ = q;
            bestIdx = static_cast<int>(i);
        }
    }

    JointActionCandidate action = rootNode->get_joint_action(bestIdx);
    string moveA = (action.moveA == Stockfish::MOVE_NONE) 
                    ? "pass" : board.uci_move(0, action.moveA);
    string moveB = (action.moveB == Stockfish::MOVE_NONE) 
                    ? "pass" : board.uci_move(1, action.moveB);
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
        if (children.empty()) {
            break;
        }
        
        // If debug log level, print all candidate moves at this PV node
        if (g_logLevel == LOG_DEBUG) {
            cout << "PV depth " << depth << " candidates:" << endl;
            for (size_t i = 0; i < children.size(); ++i) {
                JointActionCandidate candAction = currentNode->get_joint_action(static_cast<int>(i));
                string candMoveA = (candAction.moveA == Stockfish::MOVE_NONE) ? "pass" : tempBoard.uci_move(0, candAction.moveA);
                string candMoveB = (candAction.moveB == Stockfish::MOVE_NONE) ? "pass" : tempBoard.uci_move(1, candAction.moveB);
                float candQ = children[i]->Q();
                int candVisits = children[i]->get_visits();
                cout << "  (" << candMoveA << ", " << candMoveB << ")"
                     << "  Q: " << std::fixed << std::setprecision(3) << candQ
                     << "  Visits: " << candVisits << endl;
            }
        }
        
        // Find child with most visits
        int bestIdx = 0;
        int maxVisits = 0;
        
        for (size_t i = 0; i < children.size(); i++) {
            int visits = children[i]->get_visits();
            if (visits > maxVisits) {
                maxVisits = visits;
                bestIdx = static_cast<int>(i);
            }
        }
        
        // Get the joint action for this move
        JointActionCandidate action = currentNode->get_joint_action(bestIdx);
        
        // Format move string
        string moveA = (action.moveA == Stockfish::MOVE_NONE) 
                        ? "pass" : tempBoard.uci_move(0, action.moveA);
        string moveB = (action.moveB == Stockfish::MOVE_NONE) 
                        ? "pass" : tempBoard.uci_move(1, action.moveB);
        
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
