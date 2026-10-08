#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <vector>

#include "environment/board.h"
#include "nn/engine.h"

namespace single_board {

struct Evaluation {
    float value = 0.0f;  // From the current side to move.
    std::vector<float> priors;
};
using Evaluator = std::function<Evaluation(Board&, const std::vector<Stockfish::Move>&)>;

struct Limits {
    size_t nodes = 0;
    int moveTimeMs = 1000;
    int maxDepth = 128;
};
struct Result {
    Stockfish::Move move = Stockfish::MOVE_NONE;
    size_t nodes = 0;
    int depth = 0;
    float value = 0.0f;
    bool mateInOne = false;
    std::vector<Stockfish::Move> pv;
};

// Rules are evaluated on board A. Board B never generates moves or feeds pieces.
// Returns a value from the side to move when the position is terminal.
bool terminal_value(Board& board, float& value);
Result search(Board& board, const Evaluator& evaluate, const Limits& limits,
              const std::atomic<bool>& stop);
Result search(Board& board, Engine& engine, const Limits& limits,
              const std::atomic<bool>& stop);

}  // namespace single_board
