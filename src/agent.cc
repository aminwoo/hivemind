#include "agent.h"
#include <iomanip>
#include <vector>
#include <mutex>
#include <thread>
#include <iostream>
#include <cassert>

using namespace std;

Agent::Agent(int numThreads)
    : numberOfThreads(numThreads), running(false) {
    mapWithMutex.hashTable.reserve(1e6);

    // Create the specified number of SearchThread instances.
    for (int i = 0; i < numberOfThreads; i++) {
        searchThreads.push_back(new SearchThread(&mapWithMutex));
    }
}

Agent::~Agent() {
    for (auto searchThread : searchThreads) {
        delete searchThread;
    }
}

void Agent::run_search(Board& board, const vector<Engine*>& engines, int moveTime, Stockfish::Color side, bool hasNullMove) {
    if (board.legal_moves(side).empty()) {
        cout << "bestmove (none)" << endl;
        return;
    }

    // Create the rootNode node and search info for the search.
    //rootNode = make_shared<Node>(side);
    //clear_table();
    NodeKey key { board.hash_key(), side, hasNullMove };
    rootNode = get_root_node_from_tree(key);
    if (rootNode == nullptr) {
        rootNode = make_shared<Node>(side);
        mapWithMutex.hashTable.insert({ key, rootNode });
    }

    SearchInfo* searchInfo = new SearchInfo(chrono::steady_clock::now(), moveTime);

    // Allocate an array of thread pointers (one per search thread).
    thread** threads = new thread*[numberOfThreads];

    // Ensure that there is at least one engine available.
    if (engines.empty()) {
        cerr << "Error: No engines available for search." << endl;
        delete searchInfo;
        delete[] threads;
        return;
    }

    // Launch search threads.
    // Here, we assign an engine to each thread in a round-robin manner.
    for (int i = 0; i < numberOfThreads; i++) {
        // Set the rootNode node and search info for the thread's search state.
        searchThreads[i]->set_root_node(rootNode.get());
        searchThreads[i]->set_search_info(searchInfo);

        // Choose an engine from the provided collection.
        Engine* engine = engines[i % engines.size()];
        
        // Create a new thread for the search.
        // Note: run_search_thread is assumed to be a function that accepts a SearchThread*,
        // a Board reference, and an Engine reference.
        threads[i] = new thread(run_search_thread, searchThreads[i], ref(board), engine, hasNullMove);
    }

    // Wait for all threads to finish.
    for (int i = 0; i < numberOfThreads; i++) {
        threads[i]->join();
    }

    // Retrieve the principal variation from the search.
    cout << setprecision(3) << fixed;
    cout << "info time " << (searchInfo->elapsed() / 1000);
    cout << " Q value " << rootNode->Q();
    cout << " nodes " << searchInfo->get_nodes_searched();
    cout << " nps " << searchInfo->get_nodes_searched() / (searchInfo->elapsed() / 1000);
    cout << " collisions " << searchInfo->get_collisions();

    vector<pair<int, Stockfish::Move>> pv = rootNode->get_principle_variation();
    cout << " pv ";
    for (auto action : pv) {
        cout << board.uci_move(action.first, action.second) << " ";
    }
    cout << endl;

    // Print the best move.
    cout << "bestmove " << board.uci_move(pv[0].first, pv[0].second) << endl;

    // Clean up dynamically allocated resources.
    delete searchInfo;
    delete[] threads;
}

shared_ptr<Node> Agent::get_root_node_from_tree(NodeKey key) {
    {
        lock_guard<mutex> lock(mapWithMutex.mtx);
        auto it = mapWithMutex.hashTable.find(key);
        if (it != mapWithMutex.hashTable.end()) {
            return it->second.lock();
        } else {
            clear_table();
            return nullptr; 
        }
    }
}


void Agent::clear_table() {
    // Clear all remaining nodes of the former rootNode node.
    mapWithMutex.hashTable.clear();
    assert(mapWithMutex.hashTable.size() == 0);
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
