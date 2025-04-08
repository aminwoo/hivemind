#include "searchthread.h"
#include "utils.h"
#include <string>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <math.h>

using namespace std;

const int batchSize = 8;

SearchThread::SearchThread(MapWithMutex* mapWithMutex) : mapWithMutex(mapWithMutex) { }

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

void SearchThread::add_trajectory_buffer() {
    trajectoryBuffers.push_back(vector<pair<Node*, int>>());
}

void SearchThread::run_iteration(vector<Board>& boards, Engine* engine, bool hasNullMove) {
    float* inputPlanes = new float[batchSize * NB_INPUT_VALUES()];
    float* valueOutput = new float[batchSize];
    float* policyOutput = new float[batchSize * NB_POLICY_VALUES()];

    vector<Node*> leafs;
    for (int i = 0; i < batchSize; i++) {
        // Create new leaf node from best trajectory
        trajectoryBuffers[i].clear();
        Node* leaf = add_leaf_node(boards[i], trajectoryBuffers[i]);
        leafs.push_back(leaf);

        // Side of team to play
        Stockfish::Color actionSide = leaf->get_action_side();
        bool sit = (actionSide == root->get_action_side()) == hasNullMove;
        // Run inference
        board_to_planes(boards[i], inputPlanes + (i * NB_INPUT_VALUES()), actionSide, sit);
    }

    if (!engine->runInference(inputPlanes, valueOutput, policyOutput)) {
        cerr << "Inference failed" << endl;
        delete[] inputPlanes;
        delete[] valueOutput;
        delete[] policyOutput;
        return;
    }

    for (int i = 0; i < batchSize; i++) {
        // Side of team to play
        Stockfish::Color actionSide = leafs[i]->get_action_side();
        bool sit = (actionSide == root->get_action_side()) == hasNullMove;

        vector<pair<int, Stockfish::Move>> actions = boards[i].legal_moves(actionSide);

        if (!actions.empty() && sit && boards[i].side_to_move(0) == boards[i].side_to_move(1)) {
            actions.emplace_back(0, Stockfish::MOVE_NULL);
        }

        // Softmax
        vector<float> priors = get_normalized_probablity(policyOutput + (i * NB_POLICY_VALUES()), actions, boards[i]);

        float value = valueOutput[i];
        if (actions.empty())
            value = -1.0f;
        else if (boards[i].is_draw())
            value = 0.0f;
        else
            expand_leaf_node(leafs[i], actions, priors);

        backup_leaf_node(boards[i], value, trajectoryBuffers[i]);
    }

    // Clean up allocated memory
    delete[] inputPlanes;
    delete[] valueOutput;
    delete[] policyOutput;
}

Node* SearchThread::add_leaf_node(Board& board, vector<pair<Node*, int>>& trajectoryBuffer) {
    Node* currentNode = root;
    shared_ptr<Node> nextNode;
    int childIdx;
    int boardNum = -1;

    while (true) {
        currentNode->lock();
 
        trajectoryBuffer.emplace_back(currentNode, boardNum);

        if (!currentNode->is_expanded()) {
            currentNode->unlock();
            break;
        }

        // get_best_child now returns a shared_ptr<Node>
        nextNode = currentNode->get_best_child();
        childIdx = currentNode->get_best_child_idx();

        pair<int, Stockfish::Move> action = currentNode->get_action(childIdx);
        if (action.second != Stockfish::MOVE_NULL) {
            board.push_move(action.first, action.second);
            boardNum = action.first;
        } else {
            boardNum = -1;
        }

        currentNode->apply_virtual_loss_to_child(childIdx);
        currentNode->unlock();

        // Update currentNode with the raw pointer for compatibility with other APIs.
        currentNode = nextNode.get();
    }

    currentNode->lock();
    if (currentNode->is_added())
        searchInfo->increment_collisions(1);
    else
        currentNode->set_is_added(true);
    currentNode->unlock();

    return currentNode;
}

void SearchThread::expand_leaf_node(Node* leaf, vector<pair<int, Stockfish::Move>> actions, vector<float> priors) {
    leaf->lock();

    if (!leaf->is_expanded()) {
        leaf->set_actions(actions);
        for (size_t i = 0; i < actions.size(); i++) {
            shared_ptr<Node> child = make_shared<Node>(~leaf->get_action_side());
            leaf->add_child(child, priors[i]);
        }
        if (!actions.empty())
            leaf->set_is_expanded(true);
    }
    leaf->get_best_child();
    leaf->apply_virtual_loss_to_child(leaf->get_best_child_idx());

    leaf->unlock();
}

void SearchThread::backup_leaf_node(Board& board, float value, vector<pair<Node*, int>>& trajectoryBuffer) {
    for (auto it = trajectoryBuffer.rbegin(); it != trajectoryBuffer.rend(); ++it) {
        Node* node = it->first;
        int boardNum = it->second;

        node->lock();

        if (node->is_expanded()) {
            node->update(node->get_best_child_idx(), value);
            node->revert_virtual_loss(node->get_best_child_idx());
        } else {
            node->update_terminal(value);
        }
        node->unlock();
        value = -value;

        if (boardNum != -1)
            board.pop_move(boardNum);
    }
}

void run_search_thread(SearchThread* t, Board& board, Engine* engine, bool hasNullMove) {
    Node* root = t->get_root_node();

    vector<Board> boards;
    for (int i = 0; i < batchSize; i++) {
        t->add_trajectory_buffer();
        boards.emplace_back(board);
    }

    for (int i = 1; i < 99999; i++) {
        t->run_iteration(boards, engine, hasNullMove);
        t->get_search_info()->increment_nodes(batchSize);

        if (!(i & 15)) {
            if (root->Q() > 0.90 ||
                t->get_search_info()->elapsed() > t->get_search_info()->get_move_time() ||
                !t->is_running()) {
                break;
            }
        }
    }
}

bool SearchThread::is_running() {
    return running;
}

void SearchThread::set_is_running(bool value) {
    running = value;
}
