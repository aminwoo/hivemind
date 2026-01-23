#include "agent.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#include "joint_action.h"
#include "search_params.h"

using namespace std;

Agent::Agent() : running(false) {
    // Create the transposition table for MCGS
    transpositionTable = std::make_unique<TranspositionTable>();
    transpositionTable->reserve(100000);  // Reserve space for ~100K positions
    
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
    if (board.legal_moves(teamSide).empty()) {
        cout << "bestmove (none)" << endl;
        return;
    }

    // Check for mate in 1 on either board before starting search
    auto legalMoves = board.legal_moves(teamSide);
    for (const auto& [boardNum, move] : legalMoves) {
        if (boardNum == 0) {
            board.make_moves(move, Stockfish::MOVE_NULL);
        } else {
            board.make_moves(Stockfish::MOVE_NULL, move);
        }
        
        if (board.is_checkmate(~teamSide)) {
            if (boardNum == 0) {
                board.unmake_moves(move, Stockfish::MOVE_NULL);
            } else {
                board.unmake_moves(Stockfish::MOVE_NULL, move);
            }
            
            string moveStr = board.uci_move(boardNum, move);
            string moveA = (boardNum == 0) ? moveStr : "pass";
            string moveB = (boardNum == 1) ? moveStr : "pass";
            
            cout << "info depth 1 score mate 1 nodes 1 nps 0 time 0" << endl;
            cout << "bestmove (" << moveA << "," << moveB << ")" << endl;
            return;
        }
        
        if (boardNum == 0) {
            board.unmake_moves(move, Stockfish::MOVE_NULL);
        } else {
            board.unmake_moves(Stockfish::MOVE_NULL, move);
        }
    }

    // Create the root node and search info
    rootNode = make_shared<Node>(teamSide, board.hash_key(teamHasTimeAdvantage));  // MCGS: store root hash
    SearchInfo searchInfo(chrono::steady_clock::now(), moveTime);
    
    // MCGS: Clear and set up transposition table for new search
    transpositionTable->clear();
    transpositionTable->insertOrGet(board.hash_key(teamHasTimeAdvantage), rootNode);

    if (engines.size() < static_cast<size_t>(SearchParams::NUM_SEARCH_THREADS)) {
        cerr << "Warning: Not enough engines for all threads. Need " 
             << SearchParams::NUM_SEARCH_THREADS << ", have " << engines.size() << endl;
    }

    // Set up all search threads with shared root node, search info, and transposition table
    for (auto* st : searchThreads) {
        st->set_root_node(rootNode.get());
        st->set_search_info(&searchInfo);
        st->set_transposition_table(transpositionTable.get());  // MCGS
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
    size_t tbhits = transpositionTable->getHits();
    int hashfull = transpositionTable->getFullness();
    
    // Convert Q-value [-1, 1] to centipawns using Lc0 tangent formula
    // cp = C * tan(k * Q), where C=180 and k=1.56 (tuned for Bughouse)
    // Higher scaling reflects the increased value of material in Bughouse
    constexpr float C = 180.0f;  // Scaling constant
    constexpr float k = 1.56f;   // Curvature constant
    float q = rootNode->Q();
    int cpScore = static_cast<int>(C * std::tan(k * q));
    
    // Output UCI info string
    cout << "info depth " << depth 
         << " score cp " << cpScore
         << " nodes " << nodes 
         << " nps " << nps
         << " hashfull " << hashfull
         << " tbhits " << tbhits
         << " time " << static_cast<int>(elapsedMs) << endl;
    
    string bestMoveStr = extract_best_move(board);
    cout << "bestmove " << bestMoveStr << endl;
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
    string moveA = (action.moveA == Stockfish::MOVE_NULL || action.moveA == Stockfish::MOVE_NONE) 
                    ? "pass" : board.uci_move(0, action.moveA);
    string moveB = (action.moveB == Stockfish::MOVE_NULL || action.moveB == Stockfish::MOVE_NONE) 
                    ? "pass" : board.uci_move(1, action.moveB);
    return "(" + moveA + "," + moveB + ")";
}

void Agent::set_is_running(bool value) {
    running = value;
}

bool Agent::is_running() {
    return running;
}
