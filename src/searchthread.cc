#include "searchthread.h"
#include "utils.h"
#include <string>
#include <iomanip>

const int batchSize = 8; 

SearchThread::SearchThread() {

}

SearchThread::~SearchThread() {

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

void SearchThread::add_trajectory_buffer() {
    trajectoryBuffers.push_back(std::vector<Node*>()); 
}

void SearchThread::run_iteration(std::vector<Board>& boards, Engine* engine) {
    float* inputPlanes = new float[batchSize * NB_INPUT_VALUES()];
    float* valueOutput = new float[batchSize];
    float* policyOutput = new float[batchSize * NB_POLICY_VALUES()];

    std::vector<Node*> leafs; 
    for (int i = 0; i < batchSize; i++) {
        // Create new leaf node from best trajectory
        trajectoryBuffers[i].clear();
        Node* leaf = add_leaf_node(boards[i], trajectoryBuffers[i]); 
        leafs.push_back(leaf);

        // Side of team to play
        Stockfish::Color actionSide = leaf->get_action_side();

        // Run inference
        board_to_planes(boards[i], inputPlanes + (i * NB_INPUT_VALUES()), actionSide, actionSide != root->get_action_side());
    }


    if (!engine->runInference(inputPlanes, valueOutput, policyOutput)) {
        std::cerr << "Inference failed" << std::endl;

        delete[] inputPlanes;
        delete[] valueOutput;
        delete[] policyOutput;
        return;
    }

    for (int i = 0; i < batchSize; i++) {
        // Side of team to play
        Stockfish::Color actionSide = leafs[i]->get_action_side();

        std::vector<std::pair<int, Stockfish::Move>> actions; 
        actions = boards[i].legal_moves(actionSide);

        if (!actions.empty() && actionSide != root->get_action_side() && boards[i].side_to_move(0) == boards[i].side_to_move(1)) { 
            actions.emplace_back(0, Stockfish::MOVE_NULL);
        }

        // Softmax 
        std::vector<float> priors = get_normalized_probablity(policyOutput + (i * NB_POLICY_VALUES()), actions, boards[i]);

        float value = valueOutput[i];
        if (actions.empty()) {
            value = -1.0f + (0.005f * leafs[i]->get_depth());
        }
        else {
            expand_leaf_node(leafs[i], actions, priors); 
        }
        
        backup_leaf_node(boards[i], value, trajectoryBuffers[i]);
    }

    // Clean up allocated memory
    delete[] inputPlanes;
    delete[] valueOutput;
    delete[] policyOutput;
}


Node* SearchThread::add_leaf_node(Board& board, std::vector<Node*>& trajectoryBuffer) {
    Node* currentNode = root;
    while (true) {
        currentNode->lock();

        trajectoryBuffer.emplace_back(currentNode); 
        if (!currentNode->get_is_expanded()) {
            currentNode->unlock();
            break; 
        }

        Node* bestChild = currentNode->get_best_child();
        currentNode->apply_virtual_loss_to_child(currentNode->get_best_child_idx()); 
        currentNode->unlock();

        currentNode = bestChild; 
        std::pair<int, Stockfish::Move> action = currentNode->get_action(); 
        if (action.second != Stockfish::MOVE_NULL) {
            board.push_move(action.first, action.second);
        }
    }

    currentNode->lock();
    if (currentNode->is_added()) {
        searchInfo->increment_colllisions(1);
    }
    else {
        currentNode->set_is_added(true); 
    }
    currentNode->unlock();

    return currentNode;
}

void SearchThread::expand_leaf_node(Node* leaf, std::vector<std::pair<int, Stockfish::Move>> actions, std::vector<float> priors) {
    leaf->lock(); 

    if (!leaf->get_is_expanded()) {
        size_t num_children = actions.size(); 
        for (size_t i = 0; i < num_children; i++) {
            std::shared_ptr<Node> child = std::make_shared<Node>(~leaf->get_action_side());
            leaf->add_child(child, actions[i], priors[i]); 
        }
        if (!actions.empty()) {
            leaf->set_is_expanded(true);
        }
    }
    leaf->get_best_child();
    leaf->apply_virtual_loss_to_child(leaf->get_best_child_idx()); 

    leaf->unlock(); 
}

void SearchThread::backup_leaf_node(Board& board, float value, std::vector<Node*>& trajectoryBuffer) {
    for (auto it = trajectoryBuffer.rbegin(); it != trajectoryBuffer.rend(); ++it) {
        Node* node = *it;

        node->lock();

        if (node->get_is_expanded()) {
            node->update(node->get_best_child_idx(), value); 
            node->revert_virtual_loss(node->get_best_child_idx()); 
        }
        else {
            node->update_terminal(value);
        }

        node->unlock(); 

        value = -value;

        if (node != root) {
            std::pair<int, Stockfish::Move> action = node->get_action(); 
            if (action.second != Stockfish::MOVE_NULL) {
                board.pop_move(action.first);
            }
        }
    }
}

void run_search_thread(SearchThread *t, Board& board, Engine* engine) {
    Node* root = t->get_root_node();

    std::vector<Board> boards; 
    for (int i = 0; i < batchSize; i++) {
        t->add_trajectory_buffer(); 
        boards.emplace_back(board);
    }

    for (int i = 0; i < 99999; i++) {
        t->run_iteration(boards, engine);
        t->get_search_info()->increment_nodes(batchSize); 

        if (!((i + 1) & 15)) {
            root->lock(); 
            if (root->Q() > 0.8 || t->get_search_info()->elapsed() > t->get_search_info()->get_move_time() || !t->is_running()) {
                root->unlock(); 
                break; 
            }
            root->unlock(); 
        }
    } 
}

bool SearchThread::is_running() {
    return running; 
}

void SearchThread::set_is_running(bool value) {
    running = value; 
}
