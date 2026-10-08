#include "search/single_board.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "common/utils.h"
#include "environment/planes.h"

namespace single_board {
namespace {

struct Node {
    Stockfish::Move move = Stockfish::MOVE_NONE;
    float prior = 0.0f;
    float sum = 0.0f;
    size_t visits = 0;
    bool expanded = false;
    bool terminal = false;
    float terminalValue = 0.0f;
    std::vector<std::unique_ptr<Node>> children;
    float q() const { return visits ? sum / visits : 0.0f; }
};

Node* select(Node& node) {
    Node* best = nullptr;
    float bestScore = -INFINITY;
    const float scale = 1.5f * std::sqrt(static_cast<float>(node.visits + 1));
    for (const auto& child : node.children) {
        // A child's values belong to the opponent.
        const float score = -child->q() + scale * child->prior / (1 + child->visits);
        if (score > bestScore) {
            bestScore = score;
            best = child.get();
        }
    }
    return best;
}

Node* most_visited(Node& node) {
    Node* best = nullptr;
    for (const auto& child : node.children) {
        if (!best || child->visits > best->visits
            || (child->visits == best->visits && child->prior > best->prior)) {
            best = child.get();
        }
    }
    return best;
}

float expand(Node& node, Board& board, const Evaluator& evaluate) {
    node.expanded = true;
    float value;
    if (terminal_value(board, value)) {
        node.terminal = true;
        node.terminalValue = value;
        return value;
    }
    const auto moves = board.legal_moves(BOARD_A);
    const Evaluation evaluation = evaluate(board, moves);
    if (evaluation.priors.size() != moves.size() || !std::isfinite(evaluation.value)) {
        throw std::runtime_error("Invalid single-board network evaluation");
    }
    float total = 0.0f;
    for (float prior : evaluation.priors) {
        if (!std::isfinite(prior) || prior < 0.0f) {
            throw std::runtime_error("Invalid single-board policy prior");
        }
        total += prior;
    }
    for (size_t i = 0; i < moves.size(); ++i) {
        auto child = std::make_unique<Node>();
        child->move = moves[i];
        child->prior = total > 0.0f ? evaluation.priors[i] / total : 1.0f / moves.size();
        node.children.push_back(std::move(child));
    }
    return std::clamp(evaluation.value, -1.0f, 1.0f);
}

}  // namespace

bool terminal_value(Board& board, float& value) {
    if (!board.is_single_board()) {
        throw std::invalid_argument("Single-board search needs a single-board variant");
    }
    return board.single_board_terminal_value(value);
}

Result search(Board& board, const Evaluator& evaluate, const Limits& limits,
              const std::atomic<bool>& stop) {
    using Clock = std::chrono::steady_clock;
    const auto deadline = limits.moveTimeMs > 0
        ? Clock::now() + std::chrono::milliseconds(limits.moveTimeMs) : Clock::time_point::max();
    Result result;
    float value;
    if (terminal_value(board, value)) {
        result.value = value;
        // A draw claim is optional at the root. UCI callers still need a
        // legal move if the server has not ended the game yet.
        if (value == 0.0f) {
            const auto moves = board.legal_moves(BOARD_A);
            if (!moves.empty()) result.move = moves.front();
        }
        return result;
    }
    const auto legal = board.legal_moves(BOARD_A);
    result.move = legal.front();  // A stop before inference still returns a legal move.
    if (stop.load()) return result;

    // A third check or king explosion can win without an orthodox checkmate.
    // Atomic's pseudo-royal checks and Antichess do not use normal checkers.
    const auto winningCandidates = board.variant == Board::Variant::ATOMIC
        || board.variant == Board::Variant::ANTICHESS ? legal : board.checking_moves(BOARD_A);
    for (Stockfish::Move move : winningCandidates) {
        if (stop.load() || Clock::now() >= deadline) break;
        board.push_move(BOARD_A, move);
        float nextValue;
        const bool mate = terminal_value(board, nextValue) && nextValue < 0.0f;
        board.pop_move(BOARD_A);
        if (mate) {
            result.move = move;
            result.value = 1.0f;
            result.mateInOne = true;
            result.pv = {move};
            return result;
        }
    }
    Node root;
    root.sum = expand(root, board, evaluate);
    root.visits = 1;
    result.nodes = 1;
    const size_t nodeLimit = limits.nodes ? limits.nodes : 100000;
    const int depthLimit = std::clamp(limits.maxDepth, 1, 128);
    while (result.nodes < nodeLimit && !stop.load() && Clock::now() < deadline) {
        // Make/unmake preserves root history without copying at each edge.
        std::vector<Node*> path{&root};
        Node* leaf = &root;
        int depth = 0;
        while (leaf->expanded && !leaf->terminal && depth < depthLimit) {
            leaf = select(*leaf);
            board.push_move(BOARD_A, leaf->move);
            path.push_back(leaf);
            ++depth;
        }
        try {
            value = leaf->terminal ? leaf->terminalValue
                : !leaf->expanded ? expand(*leaf, board, evaluate) : leaf->q();
        } catch (...) {
            for (int i = 0; i < depth; ++i) board.pop_move(BOARD_A);
            throw;
        }
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            ++(*it)->visits;
            (*it)->sum += value;
            value = -value;
        }
        for (int i = 0; i < depth; ++i) board.pop_move(BOARD_A);
        ++result.nodes;
        result.depth = std::max(result.depth, depth);
    }
    Node* best = most_visited(root);
    if (best) {
        result.move = best->move;
        result.value = -best->q();
        for (int i = 0; best && i < 20; ++i) {
            result.pv.push_back(best->move);
            best = most_visited(*best);
        }
    }
    return result;
}

Result search(Board& board, Engine& engine, const Limits& limits,
              const std::atomic<bool>& stop) {
    const size_t batch = engine.getBatchSize();
    std::vector<float> input(batch * NB_INPUT_VALUES(), 0.0f);
    std::vector<float> values(batch), policyA(batch * NB_POLICY_VALUES());
    std::vector<float> policyB(batch * NB_POLICY_VALUES()), wdl(batch * 3), movesLeft(batch);
    const Evaluator evaluate = [&](Board& position, const std::vector<Stockfish::Move>& moves) {
        board_to_planes(position, input.data(), position.side_to_move(BOARD_A), false);
        if (!engine.runInference(input.data(), values.data(), policyA.data(), policyB.data(),
                                 wdl.data(), movesLeft.data())) {
            throw std::runtime_error("Single-board neural inference failed");
        }
        std::vector<int> indices;
        for (Stockfish::Move move : moves) {
            int index = get_fast_policy_index(move, position.side_to_move(BOARD_A));
            // The shared network has queen/knight promotions only. Keep rook
            // and bishop promotions legal, with the queen promotion's prior.
            if (index < 0 && Stockfish::type_of(move) == Stockfish::PROMOTION) {
                index = get_fast_policy_index(
                    Stockfish::make<Stockfish::PROMOTION>(Stockfish::from_sq(move),
                        Stockfish::to_sq(move), Stockfish::QUEEN), position.side_to_move(BOARD_A));
            }
            indices.push_back(index);
        }
        return Evaluation{values[0], get_normalized_probability(policyA.data(), indices)};
    };
    return search(board, evaluate, limits, stop);
}

}  // namespace single_board
