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

void SearchThread::run_iteration(std::vector<Bugboard>& boards) {
    std::vector<Node*> leafNodes; 
    for (int i = 0; i < network.get_batch_size(); i++) {
        add_leaf_node(boards[i], leafNodes); 
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
            expand_leaf_node(leafNodes[i], boards[i], actions[i], priors, sides[i]); 
        }
        backup_leaf_node(leafNodes[i], boards[i], value);
    }
}

void SearchThread::backup_leaf_node(Node* leaf, Bugboard& board, float value) {
    int turnFactor = -1; 
    Node* curr = leaf;
    Stockfish::Color prev_action_side = curr->get_action_side(); 

    while (true) {
        curr->increment_visits(); 
        if (curr->get_parent() == nullptr) {
            break; 
        }

        Stockfish::Color next_action_side = curr->get_parent()->get_action_side(); 
        curr->add_value(value * turnFactor); 
        
        std::pair<int, Stockfish::Move> action = curr->get_action(); 
        /*if (value == -1) {
            std::cout << board.uci_move(action.first, action.second) << ' ' << turnFactor << ' ' << prev_action_side << ' ' << next_action_side << ' ' << std::endl; 
        }*/

        if (prev_action_side != next_action_side) {
            turnFactor *= -1;
        }

        prev_action_side = next_action_side; 

        //curr->add_value(value * turnFactor); 
        curr->remove_virtual_loss(1.0f); 
        curr = curr->get_parent(); 

        //turnFactor *= -1;

        board.undo_move(action.first); 
    }
}

void SearchThread::add_leaf_node(Bugboard& board, std::vector<Node*>& leafNodes) {
    Node* curr = root;
    while (true) {
        if (!curr->get_expanded()) {
            break; 
        }
        curr = curr->get_best_child();
        curr->apply_virtual_loss(1.0f); 
        std::pair<int, Stockfish::Move> action = curr->get_action(); 
        board.do_move(action.first, action.second);
    }
    curr->set_added(true); 
    leafNodes.emplace_back(curr); 
}

void SearchThread::expand_leaf_node(Node* leaf, Bugboard& board, std::vector<std::pair<int, Stockfish::Move>> actions, DynamicVector<float> priors, Stockfish::Color action_side) {
    if (leaf->get_expanded()) {
        return; 
    }
    size_t num_children = actions.size(); 
    for (size_t i = 0; i < num_children; i++) {
        std::shared_ptr<Node> child = std::make_shared<Node>(leaf, action_side);
        /*if (board.uci_move(actions[i].first, actions[i].second) == "Q@h1") {
            std::cout << "side " << action_side << std::endl; 
        }*/
        //child->set_action_side(action_side);
        child->set_prior(priors[i]);
        child->set_action(actions[i]);
        child->set_depth(leaf->get_depth() + 1);
        searchInfo->set_max_depth(child->get_depth());
        leaf->add_child(child); 
    }
    if (!actions.empty()) {
        leaf->set_expanded(true);
    }
}

void run_search_thread(SearchThread *t, Bugboard& board) {
    std::vector<Bugboard> boards; 
    for (int i = 0; i < t->network.get_batch_size(); i++) {
        boards.emplace_back(board);
    }

    for (int i = 0; i < 10000; i++) {
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