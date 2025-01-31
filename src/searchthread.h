#ifndef SEARCHTHREAD_H
#define SEARCHTHREAD_H

#include "board.h"
#include "node.h"
#include "searchinfo.h"
#include "engine.h"
#include "planes.h"
#include <chrono> 
#include <mutex>
#include <math.h>


class Node;

class SearchThread {
public: 
    SearchThread(); 
    ~SearchThread(); 
    Node* get_root_node(); 
    SearchInfo* get_search_info();
    void set_search_info(SearchInfo* info);
    void set_root_node(Node* node);
    void set_is_running(bool value); 
    Node* add_leaf_node(Board& board, std::vector<Node*>& trajectoryBuffer); 
    void expand_leaf_node(Node* leaf, std::vector<std::pair<int, Stockfish::Move>> actions, std::vector<float> priors);
    void backup_leaf_node(Board& board, float value, std::vector<Node*>& trajectoryBuffer);
    void run_iteration(Board& board, Engine& engine);
    bool is_running(); 

private: 
    Node* root; 
    bool running = false; 
    SearchInfo* searchInfo;
    std::vector<Node*> trajectoryBuffer; 

};

void run_search_thread(SearchThread *t, Board& board, Engine& engine);

#endif
