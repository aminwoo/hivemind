#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

class Engine;

/**
 * @brief Search-coverage diagnostic: for positions from a FEN file (the
 * `gennnue --fens true` format), runs an independent MCTS search at each node
 * budget and writes every root child (both boards' moves, joint prior,
 * visits, Q, and whether each move is a capture, drop, sit or forced wait)
 * as one JSON line per position. Used to tell whether a deeper search's
 * choice was even a candidate at a smaller budget.
 */
struct SearchDiagConfig {
    std::filesystem::path fenFile;
    std::filesystem::path outputFile;
    std::vector<size_t> budgets = {50, 200, 800};
    size_t positions = 5000;
    size_t every = 1;  // take every n-th line
};

int run_search_diag(Engine& engine, const SearchDiagConfig& config);
