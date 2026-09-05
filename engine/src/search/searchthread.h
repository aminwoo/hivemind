#pragma once

#include <array>
#include <unordered_set>

#include "environment/board.h"
#include "environment/joint_action.h"
#include "environment/planes.h"
#include "nn/engine.h"
#include "search/node.h"
#include "search/searchinfo.h"
#include "search/transposition_table.h"

using namespace std;

class Node;

enum class TerminalOutcome : uint8_t {
    NONE,
    WIN,
    LOSS,
    DRAW,
};

/** One representative reply-and-mate suffix for a waiting-board mate proof. */
struct WaitingMateContinuation {
    int activeBoard = -1;
    Stockfish::Move reply = Stockfish::MOVE_NONE;
    int waitingBoard = -1;
    Stockfish::Move matingMove = Stockfish::MOVE_NONE;
};

/**
 * @param partnerBoardAgnostic Classify without reading the waiting board:
 *        assume every interposable check can be blocked with a piece from the
 *        partner, and skip the waiting-board mate. Both only withdraw terminal
 *        verdicts, so a proof built on this model also holds for any other
 *        state of the waiting board - which lets a caller reuse one proof
 *        across positions that differ only there.
 * @param allowMatedTeamToMove At a live down-time root, defer a LOSS when the
 *        nominally mated team still has a legal move on its partner board. This
 *        permits one neural expansion to rank that seat's moves; descendants
 *        use normal terminal adjudication.
 */
TerminalOutcome classify_terminal_position(Board& board,
                                             Stockfish::Color teamToPlay,
                                             Stockfish::Color rootTeam,
                                             bool rootTeamHasTimeAdvantage,
                                             const std::array<int, 2>& boardSearchPlies,
                                             int* endInPly = nullptr,
                                             bool partnerBoardAgnostic = false,
                                             bool allowMatedTeamToMove = false,
                                             WaitingMateContinuation* waitingMate = nullptr);

TerminalOutcome classify_terminal_position(Board& board,
                                             Stockfish::Color teamToPlay,
                                             Stockfish::Color rootTeam,
                                             bool rootTeamHasTimeAdvantage,
                                             int searchPly,
                                             int* endInPly = nullptr,
                                             bool partnerBoardAgnostic = false,
                                             bool allowMatedTeamToMove = false,
                                             WaitingMateContinuation* waitingMate = nullptr);

/**
 * @brief Entry in an MCTS selection trajectory.
 * 
 * Stores the node, action taken, and selected child index for backup.
 */
struct TrajectoryEntry {
    Node* node;
    std::shared_ptr<Node> nodeOwner;
    JointActionCandidate action;
    int selectedChildIdx;  // Index of the child that was selected (-1 for root/leaf)
    
    TrajectoryEntry(Node* n, const JointActionCandidate& a, int idx = -1)
        : node(n), action(a), selectedChildIdx(idx) {}

    TrajectoryEntry(std::shared_ptr<Node> n, const JointActionCandidate& a, int idx = -1)
        : node(n.get()), nodeOwner(std::move(n)), action(a), selectedChildIdx(idx) {}
};

/**
 * @brief Stores context for a single leaf node in a minibatch.
 * 
 * Used to track the trajectory, board state, and leaf node for each
 * position in the inference batch so we can properly process results.
 */
struct LeafContext {
    std::shared_ptr<Node> leaf;
    vector<TrajectoryEntry> trajectory;
    vector<Stockfish::Move> actionsA;
    vector<Stockfish::Move> actionsB;
    vector<uint8_t> capturesA;
    vector<uint8_t> capturesB;
    vector<float> supplyWeightsA;
    vector<float> supplyWeightsB;
    float supplyPressure = 0.0f;
    vector<int> policyIndicesA;
    vector<int> policyIndicesB;
    Stockfish::Color teamToPlay;
    bool sitPlaneActive;
    bool boardAOnTurn;
    bool boardBOnTurn;
    bool isTerminal;     // True if this is a terminal node (draw/checkmate)
    bool hasEvaluationReservation;
    float terminalValue; // Value to use for terminal nodes
    bool isTransposition; // True if this leaf was found via transposition table (MCGS)
    uint64_t leafHash;    // Hash of the leaf position for transposition lookup
    
    LeafContext() : leaf(nullptr), teamToPlay(Stockfish::WHITE), 
                    sitPlaneActive(false), boardAOnTurn(false), boardBOnTurn(false),
                    isTerminal(false), hasEvaluationReservation(false),
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
    std::shared_ptr<Node> leaf;
    bool hasEvaluationReservation = false;
    std::shared_ptr<Node> pendingEvaluation;
    std::shared_ptr<Node> exhaustedSubtree;
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
        std::shared_ptr<Node> pendingEvaluation;
    };

    Node* root;
    std::weak_ptr<Node> rootOwner;
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
    /** Waits on a pending evaluation, bounded by the move deadline. */
    bool wait_for_pending_evaluation(Node& node);

    void collect_batch(SearchBatch& batch, Board& board,
                       bool teamHasTimeAdvantage, bool allowReservationWait);
    void process_batch(SearchBatch& batch, bool teamHasTimeAdvantage,
                       const Engine::HalfInferenceOutputs& inferenceOutputs);
    void abort_batch(SearchBatch& batch);
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
    void set_root_node(const std::shared_ptr<Node>& node);
    void set_transposition_table(TranspositionTable* table);
    void set_runtime_config(const SearchParams::RuntimeConfig& config);
    void set_inference_worker_index(size_t workerIndex);
    
    // MCGS (Monte Carlo Graph Search) with prior-ordered joint action expansion
    LeafSelection select_and_expand(
        Board& board,
        bool teamHasTimeAdvantage,
        const std::unordered_set<const Node*>* blockedNodes = nullptr);
    void expand_leaf_node(Node* leaf, 
                          const vector<Stockfish::Move>& actionsA,
                          const vector<Stockfish::Move>& actionsB,
                          const vector<float>& priorsA,
                          const vector<float>& priorsB,
                          bool teamHasTimeAdvantage,
                          bool boardAOnTurn,
                          bool boardBOnTurn,
                          const vector<uint8_t>& capturesA,
                          const vector<uint8_t>& capturesB,
                          const vector<float>& jointFactorsA = {},
                          const vector<float>& jointFactorsB = {},
                          size_t jointFactorRank = 0,
                          uint64_t positionHash = 0);
    void backup(vector<TrajectoryEntry>& trajectory, float value);
    void cancel_virtual_losses(const vector<TrajectoryEntry>& trajectory);
    
    // Minibatch MCGS - collects SearchParams::BATCH_SIZE leaves, runs batched inference, processes results
    void run_iteration(Board& board, Engine* engine, bool teamHasTimeAdvantage);
    void finish_pending_iteration(Board& board, Engine* engine,
                                  bool teamHasTimeAdvantage);
    void discard_pending_iteration(Board& board, Engine* engine);
};
