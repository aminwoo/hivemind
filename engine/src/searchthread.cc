#include "searchthread.h"
#include "utils.h"
#include "joint_action.h"
#include <string>
#include <iomanip>
#include <chrono>
#include <math.h>

using namespace std;

SearchThread::SearchThread() {
    // Pre-allocate inference buffers
    obs = new float[BATCH_SIZE * NB_INPUT_VALUES()];
    value = new float[BATCH_SIZE];
    piA = new float[BATCH_SIZE * NB_POLICY_VALUES()];
    piB = new float[BATCH_SIZE * NB_POLICY_VALUES()];
}

SearchThread::~SearchThread() {
    delete[] obs;
    delete[] value;
    delete[] piA;
    delete[] piB;
}

SearchInfo* SearchThread::get_search_info() {
    return searchInfo;
}

void SearchThread::set_search_info(SearchInfo* info) {
    searchInfo = info;
}

void SearchThread::set_root_node(Node* node) {
    root = node;
    root->set_value(0.0f);
}

Node* SearchThread::get_root_node() {
    return root;
}

void SearchThread::backup(Board& board, float value) {
    // Process nodes in reverse order (from leaf to root)
    for (auto it = trajectoryBuffer.rbegin(); it != trajectoryBuffer.rend(); ++it) {
        Node* node = it->first;

        if (node->is_expanded()) {
            node->update(node->get_best_child_idx(), value);
        } else {
            node->update_terminal(value);
        }
        value = -value;
    }
    
    // Undo all moves made during selection (skip root which has no incoming action)
    for (auto it = trajectoryBuffer.rbegin(); it != trajectoryBuffer.rend(); ++it) {
        const JointActionCandidate& action = it->second;
        if (action.moveA != Stockfish::MOVE_NONE || action.moveB != Stockfish::MOVE_NONE) {
            board.unmake_moves(action.moveA, action.moveB);
        }
    }
}

// =====================================================
// Single-Threaded MCTS with Progressive Widening
// =====================================================

/**
 * @brief Runs a single MCTS iteration using joint action progressive widening.
 * 
 * This implements the dual-head architecture for Bughouse where:
 * - The action space is the Cartesian product of moves on Board A and Board B
 * - Progressive widening limits breadth to focus on depth
 * - Joint prior P(a|s) = P_A(a_A|s) * P_B(a_B|s)
 */
void SearchThread::run_iteration(Board& board, Engine* engine, bool teamHasTimeAdvantage) {
    // Select and expand using progressive widening
    trajectoryBuffer.clear();
    Node* leaf = select_and_expand(board);

    // Check for draw (50-move rule, threefold repetition, insufficient material)
    if (board.is_draw()) {
        backup(board, 0.0f);
        return;
    }

    // Get the action side for this leaf
    Stockfish::Color actionSide = leaf->get_action_side();
    
    // For neural network input: sitting plane is active when team has time advantage
    // and it's the same side as root (our team's perspective)
    bool sitPlaneActive = (actionSide == root->get_action_side()) == teamHasTimeAdvantage;

    // Run neural network inference
    board_to_planes(board, obs, actionSide, sitPlaneActive);

    if (!engine->runInference(obs, value, piA, piB)) {
        cerr << "Inference failed" << endl;
        backup(board, 0.0f);
        return;
    }

    vector<Stockfish::Move> actionsA;
    vector<Stockfish::Move> actionsB;
    
    // Get moves for our team's color on each board
    // Check for checkmate: if it's our turn and we have no moves while in check
    if (board.side_to_move(0) == actionSide) {
        actionsA = board.legal_moves(0);
        if (actionsA.empty() && board.is_in_check(0)) {
            backup(board, -1.0f);  // Checkmate - we lose
            return;
        }
    }
    if (board.side_to_move(1) == ~actionSide) {
        actionsB = board.legal_moves(1);
        if (actionsB.empty() && board.is_in_check(1)) {
            backup(board, -1.0f);  // Checkmate - we lose
            return;
        }
    }

    vector<float> priorsA;
    vector<float> priorsB;
    
    if (actionsA.empty()) {
        actionsA.push_back(Stockfish::MOVE_NULL);
        priorsA.push_back(1.0f);
    } else {
        actionsA.push_back(Stockfish::MOVE_NULL);
        priorsA = get_normalized_probability(piA, actionsA, 0, board);
    }
    
    if (actionsB.empty()) {
        actionsB.push_back(Stockfish::MOVE_NULL);
        priorsB.push_back(1.0f);
    } else {
        actionsB.push_back(Stockfish::MOVE_NULL);
        priorsB = get_normalized_probability(piB, actionsB, 1, board);
    }

    // Expand and backup
    expand_leaf_node(leaf, actionsA, actionsB, priorsA, priorsB, teamHasTimeAdvantage);
    backup(board, value[0]);
}

/**
 * @brief Selects a leaf node and expands using progressive widening.
 */
Node* SearchThread::select_and_expand(Board& board) {
    Node* currentNode = root;
    shared_ptr<Node> nextNode;
    int childIdx;

    // Root node has no incoming action
    trajectoryBuffer.emplace_back(currentNode, JointActionCandidate());

    while (true) {
        // If not expanded, this is a leaf node
        if (!currentNode->is_expanded()) {
            break;
        }

        // Check if we should expand a new child (progressive widening)
        if (currentNode->should_expand_new_child()) {
            nextNode = currentNode->expand_next_joint_child();
            if (nextNode) {
                childIdx = currentNode->get_expanded_count() - 1;
                currentNode->set_best_child_idx(childIdx);
                
                JointActionCandidate action = currentNode->get_joint_action(childIdx);
                board.make_moves(action.moveA, action.moveB);
                trajectoryBuffer.emplace_back(nextNode.get(), action);
                
                // Return the newly expanded leaf
                return nextNode.get();
            }
        }

        // Standard PUCT selection among expanded children
        nextNode = currentNode->get_best_expanded_child();
        if (!nextNode) {
            break;  // No children available
        }
        childIdx = currentNode->get_best_child_idx();

        JointActionCandidate action = currentNode->get_joint_action(childIdx);
        board.make_moves(action.moveA, action.moveB);
        
        trajectoryBuffer.emplace_back(nextNode.get(), action);
        currentNode = nextNode.get();
    }

    return currentNode;
}

/**
 * @brief Expands a leaf node with joint action candidates using lazy priority queue.
 * @param teamHasTimeAdvantage If true, team is up on time and can sit when on turn
 */
void SearchThread::expand_leaf_node(Node* leaf,
                                    const vector<Stockfish::Move>& actionsA,
                                    const vector<Stockfish::Move>& actionsB,
                                    const vector<float>& priorsA,
                                    const vector<float>& priorsB,
                                    bool teamHasTimeAdvantage) {
    if (!leaf->is_expanded()) {
        leaf->init_joint_generator(actionsA, actionsB, priorsA, priorsB, teamHasTimeAdvantage);
        
        if (leaf->has_joint_candidates()) {
            shared_ptr<Node> firstChild = leaf->expand_next_joint_child();
            if (firstChild) {
                leaf->set_is_expanded(true);
                leaf->set_best_child_idx(0);
            }
        }
    } else {
        leaf->get_best_expanded_child();
    }
}
