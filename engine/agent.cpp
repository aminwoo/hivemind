#include "agent.h"


Agent::Agent() {
    for (int i = 0; i < numberOfThreads; i++) {
        searchThreads.emplace_back(new SearchThread);
    }
}

Agent::~Agent() {
    for (auto searchThread : searchThreads) {
        delete searchThread;
    }
}

void Agent::run_search(Bugboard& board, int move_time) {
    auto start = std::chrono::steady_clock::now();

    Stockfish::Color side = board.side_to_move(0); 
    if (side == board.side_to_move(1)) { // Diagonal players on turn. Team with less time moves first.
        Clock clock = board.get_clock(); 
        if (clock.get_time(0, side) > clock.get_time(1, side)) {
            side = ~side; 
        }
    }  
    
    Node* root = new Node(side);
    SearchInfo* searchInfo = new SearchInfo(start, move_time);
    std::thread** threads = new std::thread*[numberOfThreads];

    for (int i = 0; i < numberOfThreads; i++) {
        searchThreads[i]->set_root_node(root);
        searchThreads[i]->set_search_info(searchInfo);
        threads[i] = new std::thread(run_search_thread, searchThreads[i],std::ref(board));
    }

    for (int i = 0; i < numberOfThreads; i++) {
        threads[i]->join();
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::vector<Node*> pv = root->get_principle_variation(); 
    std::cout << std::setprecision(4) << std::fixed;
    std::cout << "info time " << diff.count();
    std::cout << " Q value " << pv[0]->Q();
    std::cout << " nodes " <<  searchInfo->get_nodes_searched();
    std::cout << " nps " <<  searchInfo->get_nodes_searched() / diff.count();
    std::cout << " collisions " << searchInfo->get_collisions();
    std::cout << " pv ";
    for (Node* node : pv) {
        std::pair<int, Stockfish::Move> action = node->get_action(); 
        std::cout << action.first << "-" << board.uci_move(action.first, action.second) << " ";

    }
    std::cout << std::endl; 

    std::pair<int, Stockfish::Move> action = pv[0]->get_action(); 
    std::cout << "bestmove " << action.first << "-" << board.uci_move(action.first, action.second) << std::endl;

    delete root; 
    delete searchInfo; 
    delete[] threads;
}

void Agent::set_is_running(bool value) {
    mtx.lock();
    running = value; 
    mtx.unlock(); 

    for (int i = 0; i < numberOfThreads; i++) {
        searchThreads[i]->set_is_running(value);
    }
}

bool Agent::is_running() {
    return running; 
}

