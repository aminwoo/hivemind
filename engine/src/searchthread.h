#pragma once

#include "board.h"
#include "node.h"
#include "searchinfo.h"
#include "engine.h"
#include "planes.h"
#include "joint_action.h"

using namespace std;

class Node;

class SearchThread {
private: 
    Node* root; 
    SearchInfo* searchInfo;
    
    // Trajectory stores (node, joint action) pairs for backup and move undoing
    vector<pair<Node*, JointActionCandidate>> trajectoryBuffer;
    
    // Pre-allocated inference buffers to avoid per-iteration allocations
    float* obs = nullptr;
    float* value = nullptr;
    float* piA = nullptr;
    float* piB = nullptr;

public: 
    SearchThread();
    ~SearchThread(); 

    Node* get_root_node(); 
    SearchInfo* get_search_info();
    void set_search_info(SearchInfo* info);
    void set_root_node(Node* node);
    
    // MCTS with joint action progressive widening
    Node* select_and_expand(Board& board);
    void expand_leaf_node(Node* leaf, 
                          const vector<Stockfish::Move>& actionsA,
                          const vector<Stockfish::Move>& actionsB,
                          const vector<float>& priorsA,
                          const vector<float>& priorsB,
                          bool teamHasTimeAdvantage);
    void backup(Board& board, float value);
    void run_iteration(Board& board, Engine* engine, bool teamHasTimeAdvantage);
};
