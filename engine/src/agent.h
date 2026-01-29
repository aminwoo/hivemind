#pragma once

#include <atomic>
#include <thread>
#include <vector>
#include "searchthread.h"
#include "board.h"
#include "node.h"
#include "engine.h"
#include "search_params.h"
#include "transposition_table.h"
#include "globals.h"
#include "joint_action.h"
#include "rl/rl_settings.h"

/**
 * @brief Search options to configure Agent::run_search behavior.
 */
struct SearchOptions {
    // Stopping conditions (one must be set)
    size_t targetNodes = 0;      // Stop after this many nodes (0 = use time)
    int moveTimeMs = 0;          // Stop after this many milliseconds (0 = use nodes)
    
    // UCI mode options
    bool verbose = false;        // Output UCI info strings (info, bestmove)
    bool checkMateIn1 = false;   // Check for immediate mate before search
    
    // Self-play exploration options  
    float dirichletAlpha = 0.0f;   // Dirichlet noise alpha (0 = no noise)
    float dirichletEpsilon = 0.0f; // Fraction of prior to replace with noise (0 = no noise)
    
    // Convenience constructors
    static SearchOptions uci(int moveTimeMs) {
        SearchOptions opts;
        opts.moveTimeMs = moveTimeMs;
        opts.verbose = true;
        opts.checkMateIn1 = true;
        return opts;
    }
    
    static SearchOptions selfplay(size_t nodes, const RLSettings& settings) {
        SearchOptions opts;
        opts.targetNodes = nodes;
        opts.verbose = false;
        opts.checkMateIn1 = false;
        opts.dirichletAlpha = settings.dirichletAlpha;
        opts.dirichletEpsilon = settings.dirichletEpsilon;
        return opts;
    }
    
    static SearchOptions selfplay(int moveTimeMs, const RLSettings& settings) {
        SearchOptions opts;
        opts.moveTimeMs = moveTimeMs;
        opts.verbose = false;
        opts.checkMateIn1 = false;
        opts.dirichletAlpha = settings.dirichletAlpha;
        opts.dirichletEpsilon = settings.dirichletEpsilon;
        return opts;
    }
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
    std::vector<SearchThread*> searchThreads;
    std::atomic<bool> running;                            
    shared_ptr<Node> rootNode;
    std::unique_ptr<TranspositionTable> transpositionTable;  // MCGS transposition table
    int numThreads;                                          // Number of search threads

public:
    /**
     * @brief Constructs a multi-threaded Agent with MCGS support.
     * @param numThreads Number of search threads (0 = use SearchParams::NUM_SEARCH_THREADS)
     */
    Agent(int numThreads = 0);

    /**
     * @brief Destructor to clean up resources.
     */
    ~Agent();

    /**
     * @brief Unified search function for both UCI and self-play modes.
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
    
    /**
     * @brief Legacy wrapper for UCI mode search. Use run_search with SearchOptions::uci() instead.
     */
    void run_search(Board& board, const std::vector<Engine*>& engines, int moveTime, 
                    Stockfish::Color side, bool teamHasTimeAdvantage);
    
    /**
     * @brief Legacy wrapper for silent search. Use run_search with SearchOptions::selfplay() instead.
     */
    JointActionCandidate run_search_silent(Board& board, const std::vector<Engine*>& engines, 
                                           size_t targetNodes, int moveTimeMs, 
                                           Stockfish::Color side, bool teamHasTimeAdvantage, 
                                           const RLSettings& settings, float temperature);

    /**
     * @brief Extracts the best move from the root node after search.
     * @param board The board state for move formatting.
     * @return String representation of the best joint move.
     */
    std::string extract_best_move(Board& board);

    /**
     * @brief Extracts the principal variation (PV) by following most-visited children.
     * @param board The current board position.
     * @param maxDepth Maximum number of moves to extract in the PV.
     * @return Space-separated sequence of joint moves.
     */
    std::string extract_pv(Board& board, int maxDepth);

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
     * @brief Get the transposition table for stats reporting.
     */
    TranspositionTable* getTranspositionTable() const {
        return transpositionTable.get();
    }
    
    /**
     * @brief Get the root node after search for extracting visit distributions.
     * Used for AlphaZero-style training data generation.
     * @return Shared pointer to the root node, or nullptr if no search has been run.
     */
    std::shared_ptr<Node> get_root_node() const {
        return rootNode;
    }
};
