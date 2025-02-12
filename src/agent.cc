#include "agent.h"
#include <iomanip>
#include <vector>
#include <mutex>


Agent::Agent(int numThreads)
    : numberOfThreads(numThreads), running(false)
{
    // Create the specified number of SearchThread instances.
    for (int i = 0; i < numberOfThreads; i++) {
        searchThreads.push_back(new SearchThread());
    }
}

Agent::~Agent() {
    for (auto searchThread : searchThreads) {
        delete searchThread;
    }
}

void Agent::run_search(Board& board, const std::vector<Engine*>& engines, int move_time) {
    // Determine which side to move.
    Stockfish::Color side = board.side_to_move(0);
    if (board.is_checkmate(side)) {
        std::cout << "bestmove (none)" << std::endl;
        return;
    }

    // Create the root node and search info for the search.
    Node* root = new Node(side);
    SearchInfo* searchInfo = new SearchInfo(std::chrono::steady_clock::now(), move_time);

    // Allocate an array of thread pointers (one per search thread).
    std::thread** threads = new std::thread*[numberOfThreads];

    // Ensure that there is at least one engine available.
    if (engines.empty()) {
        std::cerr << "Error: No engines available for search." << std::endl;
        delete root;
        delete searchInfo;
        delete[] threads;
        return;
    }

    // Launch search threads.
    // Here, we assign an engine to each thread in a round-robin manner.
    for (int i = 0; i < numberOfThreads; i++) {
        // Set the root node and search info for the thread's search state.
        searchThreads[i]->set_root_node(root);
        searchThreads[i]->set_search_info(searchInfo);

        // Choose an engine from the provided collection.
        Engine* engine = engines[i % engines.size()];
        
        // Create a new thread for the search.
        // Note: run_search_thread is assumed to be a function that accepts a SearchThread*,
        // a Board reference, and an Engine reference.
        threads[i] = new std::thread(run_search_thread, searchThreads[i], std::ref(board), engine);
    }

    // Wait for all threads to finish.
    for (int i = 0; i < numberOfThreads; i++) {
        threads[i]->join();
    }

    // Retrieve the principal variation from the search.
    std::vector<Node*> pv = root->get_principle_variation(); 
    std::cout << std::setprecision(3) << std::fixed;
    std::cout << "info time " << (searchInfo->elapsed() / 1000);
    std::cout << " Q value " << -pv[0]->Q();
    std::cout << " nodes " <<  searchInfo->get_nodes_searched();
    std::cout << " nps " <<  searchInfo->get_nodes_searched() / (searchInfo->elapsed() / 1000);
    std::cout << " collisions " << searchInfo->get_collisions();
    std::cout << " pv ";
    for (Node* node : pv) {
        std::pair<int, Stockfish::Move> action = node->get_action(); 
        std::cout << board.uci_move(action.first, action.second) << " ";
    }
    std::cout << std::endl; 

    // Print the best move.
    std::pair<int, Stockfish::Move> action = pv[0]->get_action(); 
    std::cout << "bestmove " << board.uci_move(action.first, action.second) << std::endl;

    // Clean up dynamically allocated resources.
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

