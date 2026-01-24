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

public:
    /**
     * @brief Constructs a multi-threaded Agent with MCGS support.
     */
    Agent();

    /**
     * @brief Destructor to clean up resources.
     */
    ~Agent();

    /**
     * @brief Runs the search operation on the given board using provided engines.
     * @param board The board on which to perform the search.
     * @param engines A vector of engine pointers to use during the search.
     * @param move_time The allotted time for move calculation.
     * @param teamHasTimeAdvantage If true, team is ahead on time and can double-sit.
     */
    void run_search(Board& board, const std::vector<Engine*>& engines, int move_time, Stockfish::Color side, bool teamHasTimeAdvantage);

    /**
     * @brief Runs a silent search for self-play (no UCI output).
     * @param board The board on which to perform the search.
     * @param engines A vector of engine pointers to use during the search.
     * @param targetNodes The number of MCTS iterations to run.
     * @param side The side to move.
     * @param teamHasTimeAdvantage If true, team is ahead on time.
     * @param settings RL settings for Dirichlet noise and temperature.
     * @param temperature Temperature for move selection (0 = greedy, >0 = stochastic).
     * @return The best joint action found.
     */
    JointActionCandidate run_search_silent(Board& board, const std::vector<Engine*>& engines, size_t targetNodes, Stockfish::Color side, bool teamHasTimeAdvantage, const RLSettings& settings, float temperature);

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
