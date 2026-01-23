#include "agent.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "joint_action.h"

using namespace std;

Agent::Agent() : running(false) {
    searchThread = new SearchThread();
}

Agent::~Agent() {
    delete searchThread;
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
    rootNode = make_shared<Node>(teamSide);
    SearchInfo searchInfo(chrono::steady_clock::now(), moveTime);

    if (engines.empty()) {
        cerr << "Error: No engines available for search." << endl;
        return;
    }

    Engine* engine = engines[0];
    searchThread->set_root_node(rootNode.get());
    searchThread->set_search_info(&searchInfo);
    running = true;

    // Run search iterations until time is up or stop is called
    while (running && searchInfo.elapsed() < moveTime) {
        searchThread->run_iteration(board, engine, teamHasTimeAdvantage);
        searchInfo.increment_nodes(1);
    }

    running = false;
    
    // Calculate search statistics
    double elapsedMs = searchInfo.elapsed();
    int nodes = searchInfo.get_nodes_searched();
    int depth = searchInfo.get_max_depth();
    int nps = (elapsedMs > 0) ? static_cast<int>((nodes * 1000.0) / elapsedMs) : 0;
    
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
