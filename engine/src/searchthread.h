#pragma once

#include <array>

#include "board.h"
#include "node.h"
#include "searchinfo.h"
#include "engine.h"
#include "planes.h"
#include "joint_action.h"
#include "transposition_table.h"

using namespace std;

class Node;

enum class TerminalOutcome : uint8_t {
    NONE,
    WIN,
    LOSS,
    DRAW,
};

TerminalOutcome classify_terminal_position(Board& board,
                                             Stockfish::Color teamToPlay,
                                             Stockfish::Color rootTeam,
                                             bool rootTeamHasTimeAdvantage,
                                             int searchPly,
                                             int* endInPly = nullptr);

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
    vector<TrajectoryEntry> trajectory;
    Stockfish::Color teamToPlay;
    bool sitPlaneActive;
    bool isTerminal;     // True if this is a terminal node (draw/checkmate)
    bool hasEvaluationReservation;
    float terminalValue; // Value to use for terminal nodes
    bool isTransposition; // True if this leaf was found via transposition table (MCGS)
    uint64_t leafHash;    // Hash of the leaf position for transposition lookup
    
    LeafContext() : leaf(nullptr), teamToPlay(Stockfish::WHITE), 
                    sitPlaneActive(false), isTerminal(false), hasEvaluationReservation(false),
                    terminalValue(0.0f),
                    isTransposition(false), leafHash(0) {}
    
    // Move constructor
    LeafContext(LeafContext&& other) noexcept = default;
    LeafContext& operator=(LeafContext&& other) noexcept = default;
    
    // Delete copy operations
    LeafContext(const LeafContext&) = delete;
    LeafContext& operator=(const LeafContext&) = delete;
};

struct LeafSelection {
    Node* leaf = nullptr;
    bool hasEvaluationReservation = false;
    Node* pendingEvaluation = nullptr;
};

class SearchThread {
private: 
    struct SearchBatch {
        __half* observations = nullptr;
        vector<LeafContext> contexts;
        int validInferenceCount = 0;
        int sameBatchCollisions = 0;
        int reservationCollisions = 0;
    };

    struct CanonicalChildResult {
        bool expanded = false;
        Node* pendingEvaluation = nullptr;
    };

    Node* root; 
    SearchInfo* searchInfo;
    TranspositionTable* transpositionTable;  // Shared across all search threads (MCGS)
    SearchParams::RuntimeConfig runtimeConfig;
    
    // Trajectory stores entries for backup and move undoing
    vector<TrajectoryEntry> trajectoryBuffer;
    
    // Double-buffered batches allow leaf collection to overlap GPU inference.
    array<SearchBatch, 2> batches;
    int pendingBatchIndex = -1;
    
    // Current batch size (initialized on first run_iteration call)
    int currentBatchSize = 0;
    size_t inferenceWorkerIndex = 0;
    
    // Allocate/reallocate buffers for given batch size
    void ensureBufferSize(int batchSize);
    void collect_batch(SearchBatch& batch, Board& board,
                       bool teamHasTimeAdvantage, bool allowReservationWait);
    void process_batch(SearchBatch& batch, Board& board,
                       bool teamHasTimeAdvantage,
                       const Engine::HalfInferenceOutputs& inferenceOutputs);
    void abort_batch(SearchBatch& batch, Board& board);
    CanonicalChildResult canonicalize_child(
        Board& board,
        Node* parent,
        int childIdx,
        const JointActionCandidate& action,
        shared_ptr<Node>& child,
        bool& hasEvaluationReservation,
        bool teamHasTimeAdvantage);

public: 
    SearchThread();
    ~SearchThread(); 

    void set_search_info(SearchInfo* info);
    void set_root_node(Node* node);
    void set_transposition_table(TranspositionTable* table);
    void set_runtime_config(const SearchParams::RuntimeConfig& config);
    void set_inference_worker_index(size_t workerIndex);
    
    // MCGS (Monte Carlo Graph Search) with prior-ordered joint action expansion
    LeafSelection select_and_expand(Board& board, bool teamHasTimeAdvantage);
    void expand_leaf_node(Node* leaf, 
                          const vector<Stockfish::Move>& actionsA,
                          const vector<Stockfish::Move>& actionsB,
                          const vector<float>& priorsA,
                          const vector<float>& priorsB,
                          bool teamHasTimeAdvantage,
                          bool boardAOnTurn,
                          bool boardBOnTurn,
                          uint64_t positionHash = 0);
    void backup(vector<TrajectoryEntry>& trajectory, 
                Board& board, float value);
    void cancel_virtual_losses(const vector<TrajectoryEntry>& trajectory);
    
    // Minibatch MCGS - collects SearchParams::BATCH_SIZE leaves, runs batched inference, processes results
    void run_iteration(Board& board, Engine* engine, bool teamHasTimeAdvantage);
    void finish_pending_iteration(Board& board, Engine* engine,
                                  bool teamHasTimeAdvantage);
};
