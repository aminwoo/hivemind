#pragma once

#include "board.h"
#include "node.h"
#include "searchinfo.h"
#include "engine.h"
#include "planes.h"
#include "joint_action.h"

using namespace std;

class Node;

/**
 * @brief Entry in an MCTS selection trajectory.
 * 
 * Stores the node, action taken, and selected child index for backup.
 */
struct TrajectoryEntry {
    Node* node;
    JointActionCandidate action;
    int selectedChildIdx;  // Index of the child that was selected (-1 for root/leaf)
    
    TrajectoryEntry(Node* n, const JointActionCandidate& a, int idx = -1)
        : node(n), action(a), selectedChildIdx(idx) {}
};

/**
 * @brief Stores context for a single leaf node in a minibatch.
 * 
 * Used to track the trajectory, board state, and leaf node for each
 * position in the inference batch so we can properly process results.
 */
struct LeafContext {
    Node* leaf;
    std::unique_ptr<Board> boardState;  // Copy of board at the leaf position
    vector<TrajectoryEntry> trajectory;
    Stockfish::Color teamToPlay;
    bool sitPlaneActive;
    bool isTerminal;     // True if this is a terminal node (draw/checkmate)
    float terminalValue; // Value to use for terminal nodes
    
    LeafContext() : leaf(nullptr), teamToPlay(Stockfish::WHITE), 
                    sitPlaneActive(false), isTerminal(false), terminalValue(0.0f) {}
    
    // Move constructor
    LeafContext(LeafContext&& other) noexcept = default;
    LeafContext& operator=(LeafContext&& other) noexcept = default;
    
    // Delete copy operations
    LeafContext(const LeafContext&) = delete;
    LeafContext& operator=(const LeafContext&) = delete;
};

class SearchThread {
private: 
    Node* root; 
    SearchInfo* searchInfo;
    
    // Trajectory stores entries for backup and move undoing
    vector<TrajectoryEntry> trajectoryBuffer;
    
    // Pre-allocated inference buffers for batched inference
    float* obs = nullptr;
    float* value = nullptr;
    float* piA = nullptr;
    float* piB = nullptr;
    
    // Batch collection for minibatch MCTS
    vector<LeafContext> batchContexts;

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
    void backup(vector<TrajectoryEntry>& trajectory, 
                Board& board, float value);
    
    // Minibatch MCTS - collects BATCH_SIZE leaves, runs batched inference, processes results
    void run_batch_iteration(Board& board, Engine* engine, bool teamHasTimeAdvantage);
    
    // Single iteration (legacy, calls run_batch_iteration internally)
    void run_iteration(Board& board, Engine* engine, bool teamHasTimeAdvantage);
};
