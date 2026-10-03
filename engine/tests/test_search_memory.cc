#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
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

namespace {

/// An item whose destructor holds a GC thread until the test releases it.
class Gate {
public:
    std::shared_ptr<void> item() {
        return std::shared_ptr<int>(new int(0), [this](int* value) {
            std::unique_lock lock(mutex_);
            ++entered_;
            cv_.notify_all();
            cv_.wait(lock, [this] { return released_; });
            delete value;
        });
    }
    bool wait_entered(int count) {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(5),
                            [&] { return entered_ >= count; });
    }
    void release() {
        std::lock_guard lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int entered_ = 0;
    bool released_ = false;
};

/// True when enqueue returns without waiting for a GC thread to free room.
bool enqueue_returns_promptly(GCThread& gc, std::shared_ptr<void> item,
                              bool critical,
                              const std::atomic<bool>* cancelled) {
    auto done = std::async(std::launch::async, [&, item]() mutable {
        gc.enqueue(std::move(item), critical, cancelled);
    });
    return done.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
}

}  // namespace

// A search on the path to a move must not wait for a backlog to drain: the
// soft cap only holds back producers that can afford to wait.
TEST(GCThreadTest, CriticalProducerPassesTheSoftCap) {
    GCThread gc;
    gc.set_workers(1);
    gc.set_capacity(1);
    gc.start();

    Gate gate;
    gc.enqueue(gate.item());
    ASSERT_TRUE(gate.wait_entered(1));  // The only worker is now held.
    auto filler = std::make_shared<Node>(Stockfish::WHITE);
    std::weak_ptr<Node> fillerObserver = filler;
    gc.enqueue(std::move(filler));      // Queue is at capacity.

    auto urgent = std::make_shared<Node>(Stockfish::WHITE);
    std::weak_ptr<Node> urgentObserver = urgent;
    EXPECT_TRUE(enqueue_returns_promptly(gc, std::move(urgent), true, nullptr));
    EXPECT_EQ(gc.pending_count(), 2u);  // Queued, not freed inline.

    gate.release();
    gc.stop();
    EXPECT_TRUE(fillerObserver.expired());
    EXPECT_TRUE(urgentObserver.expired());
}

// A background search is stopped for the position the next search needs, so
// its stop flag must end a wait for room - and the item still gets freed.
TEST(GCThreadTest, CancelledWaitStillQueuesTheItem) {
    GCThread gc;
    gc.set_workers(1);
    gc.set_capacity(1);
    gc.start();

    Gate gate;
    gc.enqueue(gate.item());
    ASSERT_TRUE(gate.wait_entered(1));
    gc.enqueue(std::make_shared<Node>(Stockfish::WHITE));

    std::atomic<bool> stopRequested{false};
    auto background = std::make_shared<Node>(Stockfish::WHITE);
    std::weak_ptr<Node> backgroundObserver = background;
    auto waiting = std::async(std::launch::async, [&, background]() mutable {
        gc.enqueue(std::move(background), false, &stopRequested);
    });
    background.reset();
    EXPECT_EQ(waiting.wait_for(std::chrono::milliseconds(50)),
              std::future_status::timeout);  // Held back by the soft cap.
    stopRequested.store(true);
    EXPECT_EQ(waiting.wait_for(std::chrono::seconds(2)),
              std::future_status::ready);
    EXPECT_EQ(gc.pending_count(), 2u);

    gate.release();
    gc.stop();
    EXPECT_TRUE(backgroundObserver.expired());
}

#if defined(__GLIBC__)
// malloc_trim locks the arenas search workers allocate from, so it must wait
// for the engine to go idle rather than run whenever the queue drains.
TEST(GCThreadTest, TrimsOnlyOnceNoSearchIsRunning) {
    GCThread gc;
    gc.start();
    {
        const GCThread::BusyScope search(gc);
        gc.enqueue(std::make_shared<Node>(Stockfish::WHITE));
        std::this_thread::sleep_for(GCThread::kIdleBeforeTrim * 3 / 2);
        EXPECT_EQ(gc.trim_count(), 0u);  // Drained, but a search is running.
    }
    const auto deadline = std::chrono::steady_clock::now()
        + GCThread::kIdleBeforeTrim * 4;
    while (gc.trim_count() == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(gc.trim_count(), 1u);  // Idle long enough: freed memory returned.
    gc.stop();
}

// With nothing freed there is nothing to return, idle or not.
TEST(GCThreadTest, DoesNotTrimWithNothingFreed) {
    GCThread gc;
    gc.start();
    { const GCThread::BusyScope search(gc); }
    std::this_thread::sleep_for(GCThread::kIdleBeforeTrim * 3 / 2);
    EXPECT_EQ(gc.trim_count(), 0u);
    gc.stop();
}
#endif

// One thread cannot keep up with a search discarding a large tree and table
// every move; the workers must free concurrently, not take turns.
TEST(GCThreadTest, WorkersFreeInParallel) {
    GCThread gc;
    gc.set_workers(2);
    gc.start();

    Gate gate;
    gc.enqueue(gate.item());
    gc.enqueue(gate.item());
    EXPECT_TRUE(gate.wait_entered(2));

    gate.release();
    gc.stop();
    EXPECT_EQ(gc.pending_count(), 0u);
}
