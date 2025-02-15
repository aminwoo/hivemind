#pragma once

#include <vector>
#include <mutex>
#include "searchthread.h"
#include "board.h"
#include "node.h"
#include "engine.h"

/**
 * @brief Manages search threads for board analysis.
 *
 * The Agent class is responsible for managing multiple search threads that
 * evaluate board positions using various engines.
 */
class Agent {
private:
    int numberOfThreads;                     ///< Number of search threads.
    std::vector<SearchThread*> searchThreads;///< Container for search threads.
    std::mutex mtx;                          ///< Mutex for thread safety.
    bool running;                            ///< Flag indicating if the agent is running.

public:
    /**
     * @brief Constructs an Agent with the specified number of threads.
     * @param numThreads Number of threads to initialize.
     */
    Agent(int numThreads);

    /**
     * @brief Destructor to clean up resources.
     */
    ~Agent();

    /**
     * @brief Runs the search operation on the given board using provided engines.
     * @param board The board on which to perform the search.
     * @param engines A vector of engine pointers to use during the search.
     * @param move_time The allotted time for move calculation.
     */
    void run_search(Board& board, const std::vector<Engine*>& engines, int move_time, Stockfish::Color side, bool canSit);

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
