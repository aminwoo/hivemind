#pragma once

#include "board.h"
#include "node.h"
#include "searchinfo.h"
#include "engine.h"
#include "planes.h"

using namespace std;

class Node;

class SearchThread {
private: 
    Node* root; 
    bool running = false; 
    MapWithMutex* mapWithMutex;
    SearchInfo* searchInfo;
    vector<vector<pair<Node*, int>>> trajectoryBuffers; 

public: 
    SearchThread(MapWithMutex* mapWithMutex); 

    Node* get_root_node(); 
    SearchInfo* get_search_info();
    void add_trajectory_buffer(); 
    void set_search_info(SearchInfo* info);
    void set_root_node(Node* node);
    void set_is_running(bool value); 
    Node* add_leaf_node(Board& board, vector<pair<Node*, int>>& trajectoryBuffer); 
    void expand_leaf_node(Node* leaf, vector<pair<int, Stockfish::Move>> actions, vector<float> priors);
    void backup_leaf_node(Board& board, float value, vector<pair<Node*, int>>& trajectoryBuffer);
    void run_iteration(vector<Board>& boards, Engine* engine, bool canSit);
    bool is_running(); 
};

void run_search_thread(SearchThread* t, Board& board, Engine* engine, bool canSit);
