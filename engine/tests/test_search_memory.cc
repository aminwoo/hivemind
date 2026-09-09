#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "search/gc_thread.h"
#include "search/node.h"
#include "search/search_params.h"

namespace {

/// Narrow enough that a handful of expansions exhausts the joint frontier.
constexpr int kWidth = 3;

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

/// Expands until the generator reports nothing left, which is what triggers
/// the frontier release inside expand_next_joint_child.
void expand_to_exhaustion(const std::shared_ptr<Node>& node,
                          const SearchParams::RuntimeConfig& config) {
    for (int guard = 0; guard < 1024; ++guard) {
        if (!node->has_unexpanded_joint_actions()) {
            return;
        }
        JointActionCandidate action;
        if (!node->expand_next_joint_child(nullptr, 0, action, config)) {
            return;
        }
    }
}

}  // namespace

// Expanding the last candidate hands the generator's frontier machinery back.
// Everything the node still exposes has to survive that, because a released
// node is an ordinary node for the rest of the search.
TEST(SearchMemoryTest, ExhaustedFrontierReleaseKeepsTheActionsReadable) {
    SearchParams::RuntimeConfig config;
    auto node = make_expanded_node(kWidth, config);
    ASSERT_NE(node, nullptr);

    expand_to_exhaustion(node, config);
    ASSERT_FALSE(node->has_unexpanded_joint_actions());

    const size_t generated = node->get_num_generated();
    ASSERT_GT(generated, 0u);
    for (size_t index = 0; index < generated; ++index) {
        const JointActionCandidate action =
            node->get_joint_action(static_cast<int>(index));
        EXPECT_NE(action.moveA, Stockfish::MOVE_NONE);
        EXPECT_NE(action.moveB, Stockfish::MOVE_NONE);
    }
}

// The release keeps the sorted move lists precisely so that a retained node
// can still be reprepared when tree reuse makes it the next root. Reaching
// that path on a released node is the regression this guards.
TEST(SearchMemoryTest, ReleasedNodeStillReparesAsAReusedRoot) {
    SearchParams::RuntimeConfig config;
    auto node = make_expanded_node(kWidth, config);
    ASSERT_NE(node, nullptr);

    expand_to_exhaustion(node, config);
    ASSERT_FALSE(node->has_unexpanded_joint_actions());

    const size_t generated = node->get_num_generated();
    std::vector<JointActionCandidate> before;
    before.reserve(generated);
    for (size_t index = 0; index < generated; ++index) {
        before.push_back(node->get_joint_action(static_cast<int>(index)));
    }

    node->configure_root_search(config, false);

    ASSERT_EQ(node->get_num_generated(), generated);
    for (size_t index = 0; index < generated; ++index) {
        const JointActionCandidate action =
            node->get_joint_action(static_cast<int>(index));
        EXPECT_EQ(action.moveA, before[index].moveA);
        EXPECT_EQ(action.moveB, before[index].moveB);
    }
}

// Back-pressure must not turn into a lost tree: whatever is handed over is
// freed, whether the thread drains it or the caller has to.
TEST(GCThreadTest, FreesEveryEnqueuedTree) {
    GCThread gc;
    gc.set_capacity(1);
    gc.start();

    std::vector<std::weak_ptr<Node>> observers;
    for (int i = 0; i < 8; ++i) {
        auto node = std::make_shared<Node>(Stockfish::WHITE);
        observers.push_back(node);
        gc.enqueue(std::move(node));
    }

    gc.stop();  // Drains before joining.
    EXPECT_EQ(gc.pending_count(), 0u);
    for (const std::weak_ptr<Node>& observer : observers) {
        EXPECT_TRUE(observer.expired());
    }
}

// Nothing drains the queue before start() or after stop(), so parking a tree
// there would strand it. The caller frees it inline instead - and must not
// block waiting for a thread that will never take it.
TEST(GCThreadTest, FreesInlineWhenNoWorkerCanDrain) {
    GCThread gc;
    gc.set_capacity(1);

    auto beforeStart = std::make_shared<Node>(Stockfish::WHITE);
    std::weak_ptr<Node> beforeStartObserver = beforeStart;
    gc.enqueue(std::move(beforeStart));
    EXPECT_TRUE(beforeStartObserver.expired());

    gc.start();
    gc.stop();

    auto afterStop = std::make_shared<Node>(Stockfish::WHITE);
    std::weak_ptr<Node> afterStopObserver = afterStop;
    gc.enqueue(std::move(afterStop));
    EXPECT_TRUE(afterStopObserver.expired());
}
