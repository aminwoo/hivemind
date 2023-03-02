#include "searchthread.h"

#include "uci.h"
#include "utils.h"

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
}

Node* SearchThread::get_root_node() {
    return root; 
}

void SearchThread::add_trajectory_buffer() {
    trajectoryBuffers.push_back(std::vector<Node*>()); 
}

void SearchThread::run_iteration(std::vector<Bugboard>& boards) {
    std::vector<Node*> leafNodes; 
    for (int i = 0; i < network.get_batch_size(); i++) {
        trajectoryBuffers[i].clear();
        add_leaf_node(boards[i], leafNodes, trajectoryBuffers[i]); 
    }

    std::vector<Stockfish::Color> sides; 
    std::vector<std::vector<std::pair<int, Stockfish::Move>>> actions; 
    for (int i = 0; i < network.get_batch_size(); i++) {
        Stockfish::Color side = boards[i].side_to_move(0); 
        if (side == boards[i].side_to_move(1)) { // Diagonal players on turn. Team with less time moves first.
            Clock clock = boards[i].get_clock(); 
            if (clock.get_time(0, side) > clock.get_time(1, side)) {
                side = ~side; 
            }
        }  
        actions.push_back(boards[i].legal_moves(side));
        network.bind_input(boards[i], side, i);
        sides.push_back(side); 
    }

    auto [valueOutputs, policyOutputs] = network.forward();
    for (int i = 0; i < network.get_batch_size(); i++) {
        DynamicVector<float> priors = get_normalized_probablity(policyOutputs[i], actions[i], boards[i]);

        float value = valueOutputs[i][0];
        if (actions[i].empty()) {
            value = -1;
        }
        else {
            expand_leaf_node(leafNodes[i], actions[i], priors, sides[i]); 
        }
        backup_leaf_node(boards[i], value, trajectoryBuffers[i]);
    }
}


void SearchThread::add_leaf_node(Bugboard& board, std::vector<Node*>& leafNodes, std::vector<Node*>& trajectoryBuffer) {
    Node* curr = root;
    while (true) {
        trajectoryBuffer.emplace_back(curr); 
        if (!curr->get_expanded()) {
            break; 
        }
        Node* bestChild = curr->get_best_child();
        curr->apply_virtual_loss_to_child(curr->get_idx(), 1.0f); 
        curr = bestChild; 
        std::pair<int, Stockfish::Move> action = curr->get_action(); 
        board.do_move(action.first, action.second);
    }
    if (curr->is_added()) {
        searchInfo->increment_colllisions(1);
    }
    else {
        curr->set_added(true); 
    }
    leafNodes.emplace_back(curr); 
}

void SearchThread::expand_leaf_node(Node* leaf, std::vector<std::pair<int, Stockfish::Move>> actions, DynamicVector<float> priors, Stockfish::Color actionSide) {
    if (!leaf->get_expanded()) {
        size_t num_children = actions.size(); 
        for (size_t i = 0; i < num_children; i++) {
            std::shared_ptr<Node> child = std::make_shared<Node>(actionSide);
            leaf->add_child(child, actions[i], priors[i]); 
        }
        if (!actions.empty()) {
            leaf->set_expanded(true);
        }
    }
    leaf->get_best_child();
    leaf->apply_virtual_loss_to_child(leaf->get_idx(), 1.0f); 
}

void SearchThread::backup_leaf_node(Bugboard& board, float value, std::vector<Node*>& trajectoryBuffer) {
    for (auto it = trajectoryBuffer.rbegin(); it != trajectoryBuffer.rend(); ++it) {
        Node* node = *it;
        if (node->get_expanded()) {
            Stockfish::Color childActionSide = node->get_child_action_side(); 
            Stockfish::Color actionSide = node->get_action_side(); 
            node->revert_virtual_loss_to_child(node->get_idx(), 1.0f); 
            node->update(node->get_idx(), value, childActionSide, actionSide); 
            if (actionSide != childActionSide) {
                value = -value; 
            }
        }
        else {
            node->update_terminal(value);
        }

        if (node != root) {
            std::pair<int, Stockfish::Move> action = node->get_action(); 
            board.undo_move(action.first);
        }
    }
}

void run_search_thread(SearchThread *t, Bugboard& board) {
    std::vector<Bugboard> boards; 
    for (int i = 0; i < t->network.get_batch_size(); i++) {
        t->add_trajectory_buffer(); 
        boards.emplace_back(board);
    }

    for (int i = 0; i < 1000; i++) {
        t->run_iteration(boards);
        t->get_search_info()->increment_nodes(t->network.get_batch_size()); 
        if (t->get_search_info()->elapsed() > t->get_search_info()->get_move_time() || !t->is_running()) {
            break; 
        }
    } 
}

bool SearchThread::is_running() {
    return running; 
}

void SearchThread::set_is_running(bool value) {
    running = value; 
}