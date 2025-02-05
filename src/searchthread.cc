#include "searchthread.h"
#include "utils.h"
#include <string>
#include <iomanip>



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

void SearchThread::run_iteration(Board& board, Engine& engine) {
    // Create new leaf node from best trajectory
    trajectoryBuffer.clear();
    Node* leaf = add_leaf_node(board, trajectoryBuffer); 

    // Side of team to play
    Stockfish::Color actionSide = leaf->get_action_side();

    // Run inference
    float* inputPlanes = new float[NB_INPUT_VALUES()];
    float* valueOutput = new float[1];
    float* policyOutput = new float[NB_POLICY_VALUES()];
    board_to_planes(board, inputPlanes, actionSide, actionSide != root->get_action_side());

    if (!engine.runInference(inputPlanes, valueOutput, policyOutput)) {
        std::cerr << "Inference failed" << std::endl;

        delete[] inputPlanes;
        delete[] valueOutput;
        delete[] policyOutput;
        return;
    }

    std::vector<std::pair<int, Stockfish::Move>> actions; 
    actions = board.legal_moves(actionSide);
    /*if (leaf == root) {*/
    /*    actions = board.legal_moves(0);*/
    /*}*/
    /*else {*/
    /*    actions = board.legal_moves(actionSide);*/
    /*}*/


    if (!actions.empty() && actionSide != root->get_action_side() && board.side_to_move(0) == board.side_to_move(1)) { 
        actions.emplace_back(0, Stockfish::MOVE_NULL);
    }

    // Softmax 
    std::vector<float> priors = get_normalized_probablity(policyOutput, actions, board);

    /*if (leaf == root) {*/
    /*    for (size_t i = 0; i < priors.size(); i++) {*/
    /*        std::cout << board.uci_move(actions[i].first, actions[i].second) << ' ' << priors[i] << std::endl;*/
    /*    }*/
    /*}*/
    /*std::cout << *board.pos[0] << std::endl;*/
    /*for (size_t i = 0; i < priors.size(); i++) {*/
    /*    std::cout << board.uci_move(actions[i].first, actions[i].second) << ' ' << priors[i] << std::endl;*/
    /*}*/

    float value = valueOutput[0];
    if (actions.empty() || board.check_mate_in_one(~actionSide)) {
        value = -1;
    }
    else {
        expand_leaf_node(leaf, actions, priors); 
    }
    
    backup_leaf_node(board, value, trajectoryBuffer);

    // Clean up allocated memory
    delete[] inputPlanes;
    delete[] valueOutput;
    delete[] policyOutput;
}


Node* SearchThread::add_leaf_node(Board& board, std::vector<Node*>& trajectoryBuffer) {
    Node* curr = root;
    while (true) {
        trajectoryBuffer.emplace_back(curr); 
        if (!curr->get_is_expanded()) {
            break; 
        }
        Node* bestChild = curr->get_best_child();
        curr = bestChild; 
        std::pair<int, Stockfish::Move> action = curr->get_action(); 
        if (action.second != Stockfish::MOVE_NULL) {
            board.push_move(action.first, action.second);
        }
    }
    if (curr->is_added()) {
        searchInfo->increment_colllisions(1);
    }
    else {
        curr->set_is_added(true); 
    }
    return curr;
}

void SearchThread::expand_leaf_node(Node* leaf, std::vector<std::pair<int, Stockfish::Move>> actions, std::vector<float> priors) {
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
}

void SearchThread::backup_leaf_node(Board& board, float value, std::vector<Node*>& trajectoryBuffer) {
    for (auto it = trajectoryBuffer.rbegin(); it != trajectoryBuffer.rend(); ++it) {
        Node* node = *it;
        if (node->get_is_expanded()) {
            node->update(node->get_best_child_idx(), value); 
        }
        else {
            node->update_terminal(value);
        }
        value = -value;

        if (node != root) {
            std::pair<int, Stockfish::Move> action = node->get_action(); 
            if (action.second != Stockfish::MOVE_NULL) {
                board.pop_move(action.first);
            }
        }
    }
}

void run_search_thread(SearchThread *t, Board& board, Engine& engine) {
    std::string bestMove = "none";

    for (int i = 0; i < 999999; i++) {
        t->run_iteration(board, engine);
        t->get_search_info()->increment_nodes(1); 
        /*Node* root = t->get_root_node();*/
        /**/
        /*if (i > 0 && i % 200 == 0) {*/
        /*    SearchInfo* searchInfo = t->get_search_info();*/
        /*    std::vector<Node*> pv = root->get_principle_variation(); */
        /**/
        /*    std::pair<int, Stockfish::Move> action = pv[0]->get_action(); */
        /*    std::string uci = board.uci_move(action.first, action.second);*/
        /**/
        /*    if (uci != bestMove) {*/
        /*        bestMove = uci;*/
        /*        std::cout << std::setprecision(3) << std::fixed;*/
        /*        std::cout << "Q value " << -pv[0]->Q();*/
        /*        std::cout << " nodes " <<  searchInfo->get_nodes_searched();*/
        /*        std::cout << " collisions " << searchInfo->get_collisions();*/
        /*        std::cout << " pv ";*/
        /*        for (Node* node : pv) {*/
        /*            std::pair<int, Stockfish::Move> action = node->get_action(); */
        /*            std::cout << board.uci_move(action.first, action.second) << " ";*/
        /**/
        /*        }*/
        /*        std::cout << std::endl; */
        /*    }*/
        /**/
        /*}*/


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
