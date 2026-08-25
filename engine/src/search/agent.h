#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>
#include "search/node.h"
#include "nn/engine.h"
#include "search/search_params.h"
#include "search/transposition_table.h"
#include "search/gc_thread.h"
#include "common/globals.h"
#include "environment/joint_action.h"

class SearchThread;
struct SearchInfo;

/**
 * @brief Search options to configure Agent::run_search behavior.
 */
struct SearchOptions {
    // Stopping conditions (one must be set)
    size_t targetNodes = 0;      // Stop after this many nodes (0 = use time)
    int moveTimeMs = 0;          // Stop after this many milliseconds (0 = use nodes)
    
    // UCI mode options
    bool verbose = false;        // Output UCI info strings (info, bestmove)
    int multiPV = 1;             // Number of principal variations to output
    bool isPonder = false;       // Ponder mode: think continuously until ponderhit or stop
    bool enablePonder = true;    // Output ponder move with bestmove
    
    SearchParams::RuntimeConfig search;
    
    // Convenience constructors
    static SearchOptions uci(int moveTimeMs, int multiPV = 1, bool isPonder = false) {
        SearchOptions opts;
        opts.moveTimeMs = moveTimeMs;
        opts.verbose = true;
        opts.multiPV = multiPV;
        opts.isPonder = isPonder;
        return opts;
    }
    
};

struct RootEdgeStats {
    JointActionCandidate action;
    int visits = 0;
};

/**
 * @brief Manages multi-threaded MCGS (Monte Carlo Graph Search) for Bughouse.
 *
 * Runs multiple search threads in parallel, each with its own engine instance.
 * All threads share the same search graph with thread-safe node operations.
 * Uses a transposition table to detect when different move sequences reach
 * the same position, enabling more efficient value estimation.
 */
class Agent {
private:
    struct RetainedRootCandidate {
        std::shared_ptr<Node> node;
        uint64_t positionHash = 0;
        std::string signature;
    };

    struct MateContinuation {
        uint64_t positionHash = 0;
        std::string signature;
        Stockfish::Color teamSide = Stockfish::WHITE;
        bool teamHasTimeAdvantage = true;
        JointActionCandidate action;
        int plyToMate = 0;
    };

    std::vector<SearchThread*> searchThreads;
    std::vector<std::thread> workerPool_;
    std::atomic<bool> running;                            
    std::mutex searchMutex_;
    std::mutex workerMutex_;
    std::condition_variable workerCv_;
    std::condition_variable workersDoneCv_;
    uint64_t workerGeneration_ = 0;
    size_t activeWorkerCount_ = 0;
    size_t completedWorkerCount_ = 0;
    bool shutdownWorkers_ = false;
    const Board* workerBoard_ = nullptr;
    std::vector<Engine*> workerEngines_;
    SearchInfo* workerSearchInfo_ = nullptr;
    bool workerTeamHasTimeAdvantage_ = false;
    size_t workerTargetNodes_ = 0;
    int workerMoveTimeMs_ = 0;
    std::exception_ptr workerException_;
    std::shared_ptr<Node> rootNode;
    std::unique_ptr<TranspositionTable> transpositionTable;  // MCGS transposition table
    int numThreads;                                          // Search threads per engine
    SearchParams::RuntimeConfig lastRuntimeConfig_;
    std::atomic<bool> isPondering_{false};                   // Whether current search is in ponder mode
    std::atomic<SearchInfo*> currentSearchInfo_{nullptr};    // Active search info pointer
    
    // Tree reuse support (CrazyAra-style). Retain every generated opponent
    // response below the selected move, not only the predicted response.
    std::vector<RetainedRootCandidate> nextRootCandidates_;
    // Exact positions and winning moves emitted by the bounded mate solver.
    // Unlike the synthetic one-edge search tree used for UCI reporting, these
    // retain every defender branch that the proof recursively verified.
    std::vector<MateContinuation> mateContinuations_;
    uint64_t lastSearchHash_ = 0;            // Hash of last search position
    
    // Garbage collection thread for async tree cleanup
    GCThread gcThread_;

    friend class AgentTreeReuseTestPeer;

    void ensure_worker_pool(size_t workerCount);
    void worker_loop(size_t workerIndex, uint64_t observedGeneration);
    void dispatch_workers(const Board& board,
                          const std::vector<Engine*>& engines,
                          SearchInfo& searchInfo,
                          bool teamHasTimeAdvantage,
                          size_t targetNodes,
                          int moveTimeMs,
                          size_t workerCount);
    void wait_for_workers();
    void reindex_reused_subtree(const std::shared_ptr<Node>& reusedRoot);
    bool try_reuse_mate_continuation(
        Board& board, Stockfish::Color teamSide, bool teamHasTimeAdvantage,
        JointActionCandidate& outAction, int& outPlyToMate) const;
    static std::string format_root_aware_uci_score(
        const std::shared_ptr<Node>& root,
        const std::shared_ptr<Node>& pvChild,
        float childQ,
        float centipawnScale = 180.0f,
        float tangentScale = 1.56f);

public:
    /**
     * @brief Constructs a multi-threaded Agent with MCGS support.
    * @param numThreads Search threads per engine (0 = use SearchParams::NUM_SEARCH_THREADS)
     */
    Agent(int numThreads = 0);

    /**
     * @brief Destructor to clean up resources.
     */
    ~Agent();

    /**
     * @brief Runs a UCI search.
     * @param board The board on which to perform the search.
     * @param engines A vector of engine pointers to use during the search.
     * @param side The side to move.
     * @param teamHasTimeAdvantage If true, team is ahead on time and can double-sit.
     * @param options Search options (stopping conditions, verbosity, noise).
     * @return The best joint action found.
     */
    JointActionCandidate run_search(Board& board, const std::vector<Engine*>& engines, 
                                    Stockfish::Color side, bool teamHasTimeAdvantage,
                                    const SearchOptions& options);

    /** Returns an immutable snapshot of the expanded root edges after search. */
    std::vector<RootEdgeStats> root_edge_stats() const;

    /** Returns the evaluated Q-value of the root node after search. */
    float root_q() const;

    /**
     * @brief Node and time budget for the root forced-mate search.
     *
     * The search runs synchronously before MCTS, so it carries a cap of its
     * own. Running out is reported as "not proven", never as a proof.
     *
     * A node count alone cannot bound it in absolute time: a joint proof node
     * enumerates board-move combinations and costs two orders of magnitude
     * more than a node of the single-board check scan. Timed searches
     * therefore also set a deadline, sampled every few probes so the clock
     * read stays negligible next to the work it guards.
     */
    struct MateSearchBudget {
        using Clock = std::chrono::steady_clock;

        uint64_t remainingNodes = SearchParams::MATE_SEARCH_NODE_BUDGET;
        bool exhausted = false;
        Clock::time_point deadline{};
        uint32_t probesSinceTimeCheck = 0;

        bool consume() {
            if (exhausted) {
                return false;
            }
            if (remainingNodes == 0) {
                exhausted = true;
                return false;
            }
            if (deadline != Clock::time_point{}
                && ++probesSinceTimeCheck
                       >= SearchParams::MATE_SEARCH_TIME_CHECK_INTERVAL) {
                probesSinceTimeCheck = 0;
                if (Clock::now() >= deadline) {
                    exhausted = true;
                    return false;
                }
            }
            --remainingNodes;
            return true;
        }

        /// Stop condition for loops that do not consume node probes.
        bool out_of_time() const {
            return deadline != Clock::time_point{} && Clock::now() >= deadline;
        }
    };

    /**
     * @brief Searches for a forced checkmate on a single board where all attacker moves are checks.
     *
     * @param budget Optional node budget; the search aborts and returns false once it is spent.
     */
    static bool search_single_board_forced_mate(
        Board& board, int boardNum, Stockfish::Color attackerColor,
        int currentPly, int maxAttackerMoves,
        Stockfish::Move& outMove, int& outPlyToMate,
        MateSearchBudget* budget = nullptr,
        bool partnerBoardAgnostic = false);

    /**
     * @brief Performs checkmate detection at the root before starting MCTS.
     */
    static bool find_root_mate(
        Board& board, Stockfish::Color teamSide, bool teamHasTimeAdvantage,
        JointActionCandidate& outAction, int& outPlyToMate,
        uint64_t nodeBudget = SearchParams::MATE_SEARCH_NODE_BUDGET,
        MateSearchBudget::Clock::time_point deadline = {});

    /**
     * @brief Proves that every legal root action permits a forced opponent mate.
     *
     * Returns the action that delays terminal loss longest. Exhausting the
     * bounded probe budget is reported conservatively as "not proven".
     */
    static bool find_root_forced_loss(
        Board& board, Stockfish::Color teamSide, bool teamHasTimeAdvantage,
        JointActionCandidate& outAction, int& outPlyToMate,
        uint64_t nodeBudget = SearchParams::MATE_SEARCH_NODE_BUDGET,
        MateSearchBudget::Clock::time_point deadline = {});
    
    /**
     * @brief Extracts PV line starting from a specific child index.
     * @param board The current board position.
     * @param childIdx The child index to start the PV from.
     * @param maxDepth Maximum number of moves to extract in the PV.
     * @return Space-separated sequence of joint moves.
     */
    std::string extract_pv_from_child(Board& board, int childIdx, int maxDepth);

    /**
     * @brief Extracts the best move from the root node after search.
     * @param board The board state for move formatting.
     * @return String representation of the best joint move.
     */
    std::string extract_best_move(Board& board);

    /**
     * @brief Extracts the predicted opponent reply from the root node after search.
     * @param board The board state for move formatting.
     * @return String representation of the predicted opponent joint move, or empty string.
     */
    std::string extract_ponder_move(Board& board);

    /**
     * @brief Extracts the principal variation (PV) by following most-visited children.
     * @param board The current board position.
     * @param maxDepth Maximum number of moves to extract in the PV.
     * @return Space-separated sequence of joint moves.
     */
    std::string extract_pv(Board& board, int maxDepth);

    /**
     * @brief Signals ponderhit from the UCI interface when the expected ponder move was played.
     * Switches search from ponder mode to active timed search, resetting search time.
     */
    void ponderhit();

    /**
     * @brief Checks if the agent is currently in ponder mode.
     * @return true if pondering, false otherwise.
     */
    bool is_pondering() const;

    /**
     * @brief Sets the running state of the agent.
     * @param value Boolean indicating whether the agent should be running.
     */
    void set_is_running(bool value);

    /**
     * @brief Checks if the agent is currently running.
     * @return true if running, false otherwise.
     */
    bool is_running();
    
    /**
     * @brief Set the hash table size in MB.
     * 
     * Resizes the transposition table used for MCGS.
     * @param sizeMB Size in megabytes (1 - 33554432)
     */
    void setHashSize(size_t sizeMB);

    /**
     * @brief Discard all search state retained between moves.
     */
    void reset_search_state();
    
    /**
     * @brief Exact positional identity of a board (both FENs, pockets included).
     *
     * Used alongside the history-sensitive search hash when adopting a retained
     * subtree, protecting against collisions and stale pocket contents.
     */
    static std::string board_signature(Board& board);

    /**
     * @brief Try to reuse the search tree from a previous search.
     *
     * Checks whether the current position matches a saved next-root candidate.
     * @param positionHash Hash of the current position.
     * @param teamSide Team to play at the current position.
     * @param signature Exact board signature of the current position.
     * @return Shared pointer to reusable root, or nullptr if no reuse possible
     */
    std::shared_ptr<Node> try_reuse_tree(uint64_t positionHash,
                                         Stockfish::Color teamSide,
                                         const std::string& signature);
    
    /**
     * @brief Store next-root candidates for tree reuse.
     * 
     * Called after search completes to retain the selected child and every
     * generated opponent response beneath it.
     */
    void store_next_root_candidates(Board& board, bool teamHasTimeAdvantage);

private:
    static bool search_single_board_forced_mate_impl(
        Board& board, int boardNum, Stockfish::Color attackerColor,
        int currentPly, int maxAttackerMoves,
        Stockfish::Move& outMove, int& outPlyToMate,
        MateSearchBudget* budget,
        std::vector<MateContinuation>* continuations,
        bool partnerBoardAgnostic = false);
    static bool find_root_mate_impl(
        Board& board, Stockfish::Color teamSide, bool teamHasTimeAdvantage,
        JointActionCandidate& outAction, int& outPlyToMate,
        uint64_t nodeBudget,
        std::vector<MateContinuation>* continuations,
        MateSearchBudget* hardBudget = nullptr,
        bool includeCaptureFeeds = true,
        MateSearchBudget::Clock::time_point deadline = {});
    
};
