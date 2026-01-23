#pragma once

#include <atomic>
#include <vector>
#include "searchthread.h"
#include "board.h"
#include "node.h"
#include "engine.h"

/**
 * @brief Manages MCTS search for Bughouse.
 *
 * Single-threaded agent that runs MCTS with joint action progressive widening.
 */
class Agent {
private:
    SearchThread* searchThread;
    std::atomic<bool> running;                            
    shared_ptr<Node> rootNode;

public:
    /**
     * @brief Constructs a single-threaded Agent.
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
     * @brief Extracts the best move from the root node after search.
     * @param board The board state for move formatting.
     * @return String representation of the best joint move.
     */
    std::string extract_best_move(Board& board);

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
};
