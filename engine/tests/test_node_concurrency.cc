#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <numeric>
#include <thread>
#include <vector>

#include "search/node.h"
#include "search/search_params.h"

namespace {

constexpr int kChildren = 20;   // The measured root width of a real search.
constexpr int kIterations = 20000;

std::vector<Stockfish::Move> synthetic_moves(int count) {
    std::vector<Stockfish::Move> moves;
    moves.reserve(count);
    for (int i = 0; i < count; ++i) {
        moves.push_back(static_cast<Stockfish::Move>(i + 1));
    }
    return moves;
}

std::vector<float> uniform_priors(int count) {
    return std::vector<float>(count, 1.0f / static_cast<float>(count));
}

/** Expands a node into `count` children, each itself expanded. */
std::shared_ptr<Node> make_expanded_node(
    int count, const SearchParams::RuntimeConfig& config) {
    auto node = std::make_shared<Node>(Stockfish::WHITE);
    node->set_depth(1);  // Not the root, so no Dirichlet noise or Gumbel state.
    const auto moves = synthetic_moves(count);
    const auto priors = uniform_priors(count);
    if (!node->try_init_and_expand(moves, moves, priors, priors,
                                   false, true, true, config)) {
        return nullptr;
    }
    return node;
}

}  // namespace

// Selection and backup run under a shared lock with atomic edge records, so a
// lost update or a torn edge would show up as visits that do not add up. Every
// iteration applies exactly one virtual loss and retires it with one visit, so
// the totals are exact regardless of how the threads interleave.
TEST(NodeConcurrencyTest, ConcurrentSelectAndBackupConserveVisits) {
    SearchParams::RuntimeConfig config;
    auto node = make_expanded_node(kChildren, config);
    ASSERT_NE(node, nullptr);
    ASSERT_TRUE(node->is_expanded());

    // Expand every child so selection never takes the evaluation-reservation
    // path, leaving the shared-lock select/backup pair as what is measured.
    const auto moves = synthetic_moves(kChildren);
    const auto priors = uniform_priors(kChildren);
    for (int i = 0; i < kChildren; ++i) {
        auto child = node->get_child(i);
        if (child) {
            child->try_init_and_expand(moves, moves, priors, priors,
                                       false, true, true, config);
        }
    }

    const unsigned threadCount = 16;
    std::atomic<int> completed{0};
    const auto started = std::chrono::steady_clock::now();

    std::vector<std::thread> workers;
    for (unsigned t = 0; t < threadCount; ++t) {
        workers.emplace_back([&, t] {
            for (int i = 0; i < kIterations; ++i) {
                Node::ChildSelection selection =
                    node->select_child_and_apply_virtual_loss(config, nullptr);
                if (!selection.child || selection.childIdx < 0) {
                    continue;
                }
                if (selection.hasEvaluationReservation) {
                    selection.child->release_evaluation_reservation();
                }
                node->update_and_remove_virtual_loss(
                    static_cast<size_t>(selection.childIdx),
                    static_cast<float>((t + i) % 3) / 2.0f - 0.5f);
                completed.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    const int done = completed.load();
    std::cout << "[ PERF     ] " << threadCount << " threads, " << done
              << " select+backup pairs in " << elapsed << " ms ("
              << (elapsed > 0 ? static_cast<long long>(done) * 1000 / elapsed : 0)
              << "/s)" << std::endl;

    const auto childVisits = node->get_child_visits();
    const long long summed = std::accumulate(
        childVisits.begin(), childVisits.end(), 0LL);
    EXPECT_EQ(summed, done) << "edge visits lost or double counted";
    EXPECT_EQ(node->get_visits(), done) << "node visits lost or double counted";
}
