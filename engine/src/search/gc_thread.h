#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "search/node.h"

/**
 * @brief Garbage collection threads for async tree cleanup.
 *
 * Frees old search tree subtrees asynchronously to avoid latency spikes
 * during time-critical search operations. When a search completes and
 * tree reuse is enabled, the unused portions of the old tree are queued
 * for async deletion by these threads.
 *
 * Based on CrazyAra's GCThread implementation.
 */
class GCThread {
public:
    /**
     * @brief Threads freeing in parallel.
     *
     * A large tree takes from half a second to several seconds to free, and a
     * search discards a tree and a transposition table every move. One thread
     * falls behind that within a few moves on a fast GPU, and every producer
     * then waits on it; several keep up.
     */
    static constexpr size_t kDefaultWorkers = 4;

    /**
     * @brief Items allowed to sit unfreed before a producer has to wait.
     *
     * A search tree is hundreds of megabytes, so an unbounded queue means a
     * producer that outruns these threads keeps every tree it discarded
     * resident at once. A discarded search hands over two items - its tree and
     * its transposition table entries - so two per worker keeps every worker
     * busy with the next search waiting, while a burst stays capped.
     */
    static constexpr size_t kDefaultCapacity = 2 * kDefaultWorkers;

    /// Least time between malloc_trim calls, which walk every arena.
    static constexpr std::chrono::seconds kTrimInterval{5};

    /**
     * @brief How long no search may have run before a trim.
     *
     * malloc_trim locks each arena while it walks it, and a search worker
     * allocating from that arena waits - on an L40S, a 237 ms trim held a
     * worker for 150 ms and the move went out 124 ms late. During a game some
     * search is nearly always running (the move, a ponder, the permanent
     * brain), so trimming waits until the engine has been idle this long.
     * Memory freed meanwhile still goes back to glibc and is reused for the
     * next tree; only returning it to the OS is deferred.
     */
    static constexpr std::chrono::milliseconds kIdleBeforeTrim{1000};

    /// Whether freed memory can be returned to the OS: malloc_trim is glibc's.
#if defined(__GLIBC__)
    static constexpr bool kCanTrim = true;
#else
    static constexpr bool kCanTrim = false;
#endif

    /// Marks a search as running for its lifetime; trimming waits for none.
    class BusyScope {
    public:
        explicit BusyScope(GCThread& gc) : gc_(gc) { gc_.begin_busy(); }
        ~BusyScope() { gc_.end_busy(); }
        BusyScope(const BusyScope&) = delete;
        BusyScope& operator=(const BusyScope&) = delete;

    private:
        GCThread& gc_;
    };

private:
    std::vector<std::thread> workers_;
    size_t workerCount_{kDefaultWorkers};
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable roomCv_;
    std::queue<std::shared_ptr<void>> deleteQueue_;
    size_t capacity_{kDefaultCapacity};
    size_t freeing_{0};  // Items taken off the queue and still being freed.
    size_t busy_{0};     // Searches running (BusyScope).
    bool trimPending_{false};  // Freed memory not yet returned to the OS.
    std::chrono::steady_clock::time_point lastBusyEnd_{};
    size_t trims_{0};
    std::atomic<bool> running_{false};
    std::atomic<bool> terminate_{false};
    std::chrono::steady_clock::time_point lastTrim_{};

    /**
     * @brief Whether to return the freed arenas to the OS now.
     *
     * Freeing a tree hands its chunks back to glibc, not to the kernel: with a
     * thread per search worker the per-thread arenas keep them, and RSS stays
     * at the high-water mark of the largest tree the process ever held. The
     * walk runs on a GC thread, but it locks each arena as it goes, so it waits
     * for everything handed over to be freed and for the engine to go idle.
     * The caller holds mutex_, so only one thread at a time decides to trim.
     */
    bool should_trim_locked(std::chrono::steady_clock::time_point now) const {
#if defined(__GLIBC__)
        return trimPending_ && deleteQueue_.empty() && freeing_ == 0
            && busy_ == 0 && now - lastBusyEnd_ >= kIdleBeforeTrim
            && now - lastTrim_ >= kTrimInterval;
#else
        (void)now;
        return false;
#endif
    }

    /**
     * @brief Worker thread loop that processes delete requests.
     */
    void worker_loop() {
        // Loops on the inner break, not on terminate_: stop() documents that
        // it waits for pending deletions, and testing the flag out here made
        // it abandon a queue that still had trees in it.
        while (true) {
            std::shared_ptr<void> nodeToDelete;
            bool trim = false;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                while (deleteQueue_.empty()
                       && !terminate_.load(std::memory_order_relaxed)) {
                    const auto now = std::chrono::steady_clock::now();
                    if (should_trim_locked(now)) {
                        trim = true;
                        trimPending_ = false;
                        lastTrim_ = now;
                        break;
                    }
                    if (trimPending_ && busy_ == 0 && freeing_ == 0) {
                        // Wake when the idle period or the trim interval ends.
                        // A thread still freeing notifies when it is done.
                        cv_.wait_until(lock, std::max(
                            lastBusyEnd_ + kIdleBeforeTrim,
                            lastTrim_ + kTrimInterval));
                    } else {
                        cv_.wait(lock);
                    }
                }

                if (!trim) {
                    if (terminate_.load(std::memory_order_relaxed) && deleteQueue_.empty()) {
                        break;
                    }

                    nodeToDelete = std::move(deleteQueue_.front());
                    deleteQueue_.pop();
                    ++freeing_;
                }
            }

            if (trim) {
#if defined(__GLIBC__)
                malloc_trim(0);
#endif
                std::lock_guard<std::mutex> lock(mutex_);
                ++trims_;
                continue;
            }

            // A producer waiting for room can proceed as soon as the slot is
            // free, which is now rather than after this item is freed.
            roomCv_.notify_all();

            // Release the node outside the lock (actual deletion happens here)
            nodeToDelete.reset();

            {
                std::lock_guard<std::mutex> lock(mutex_);
                --freeing_;
                // Without malloc_trim no trim is ever due. A pending one would
                // never clear, and the idle wait above would spin on a
                // deadline already past, holding mutex_ so a search starting
                // on this agent (BusyScope) could wait on it forever.
                trimPending_ = kCanTrim;
            }
            // Whichever thread is idle re-evaluates whether a trim is due.
            cv_.notify_one();
        }
    }

    void begin_busy() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++busy_;
    }

    void end_busy() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (busy_ > 0 && --busy_ == 0) {
                lastBusyEnd_ = std::chrono::steady_clock::now();
            }
        }
        cv_.notify_one();
    }

public:
    GCThread() = default;

    ~GCThread() {
        stop();
    }

    // Non-copyable
    GCThread(const GCThread&) = delete;
    GCThread& operator=(const GCThread&) = delete;

    /**
     * @brief Start the garbage collection threads.
     */
    void start() {
        if (running_.load(std::memory_order_relaxed)) return;

        terminate_.store(false, std::memory_order_relaxed);
        running_.store(true, std::memory_order_relaxed);
        for (size_t i = 0; i < workerCount_; ++i) {
            workers_.emplace_back(&GCThread::worker_loop, this);
        }
    }

    /**
     * @brief Stop the garbage collection threads.
     * Waits for all pending deletions to complete.
     */
    void stop() {
        if (!running_.load(std::memory_order_relaxed)) return;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            terminate_.store(true, std::memory_order_relaxed);
        }
        cv_.notify_all();
        // Release any producer parked for a slot; it frees its own tree once
        // it sees the termination flag.
        roomCv_.notify_all();

        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();

        running_.store(false, std::memory_order_relaxed);
    }

    /**
     * @brief Queue a node subtree, or any other discarded object, for async
     * deletion.
     *
     * A subtree's node and all its children are deleted asynchronously; so are
     * detached transposition table entries.
     *
     * Freeing off-thread hides the latency; it must not also hide how much is
     * still waiting to be freed. If these threads have fallen behind, the
     * producer waits for a slot rather than letting another whole tree stay
     * resident - except where waiting would cost a move:
     *
     * @param node The object to release
     * @param critical The caller is on the path to a move. It waits only once
     *        the queue holds twice its capacity, so a backlog costs memory
     *        rather than clock time while a hard cap still holds.
     * @param cancelled When set, ends a wait for room early. A background
     *        search passes its stop flag so it can always be stopped promptly.
     *        A cancelled wait still queues the object: freeing it inline is
     *        the very cost this thread exists to hide.
     */
    void enqueue(std::shared_ptr<void> node,
                 bool critical = false,
                 const std::atomic<bool>* cancelled = nullptr) {
        if (!node) return;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (running_.load(std::memory_order_relaxed)) {
                const size_t limit = critical ? 2 * capacity_ : capacity_;
                const auto has_room = [&] {
                    return deleteQueue_.size() < limit
                        || terminate_.load(std::memory_order_relaxed)
                        || (cancelled
                            && cancelled->load(std::memory_order_acquire));
                };
                // Nothing signals roomCv_ when the cancel flag is raised, so
                // a cancellable wait polls it.
                while (!has_room()) {
                    if (cancelled) {
                        roomCv_.wait_for(lock, std::chrono::milliseconds(1));
                    } else {
                        roomCv_.wait(lock);
                    }
                }
            }
            if (terminate_.load(std::memory_order_relaxed)
                || !running_.load(std::memory_order_relaxed)) {
                // Nothing will drain the queue. Free it here instead of
                // parking a tree that would outlive the threads meant to
                // collect it.
                lock.unlock();
                node.reset();
                return;
            }
            deleteQueue_.push(std::move(node));
        }
        cv_.notify_one();
    }

    /**
     * @brief Set how many items may wait unfreed before enqueue() blocks.
     * @param capacity Queue depth; clamped to at least one.
     */
    void set_capacity(size_t capacity) {
        std::lock_guard<std::mutex> lock(mutex_);
        capacity_ = std::max<size_t>(1, capacity);
        roomCv_.notify_all();
    }

    /**
     * @brief Set how many threads free in parallel. Takes effect on start().
     * @param workers Thread count; clamped to at least one.
     */
    void set_workers(size_t workers) {
        std::lock_guard<std::mutex> lock(mutex_);
        workerCount_ = std::max<size_t>(1, workers);
    }

    /**
     * @brief Get the number of pending deletions.
     */
    size_t pending_count() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));
        return deleteQueue_.size();
    }

    /**
     * @brief How many times freed memory has been returned to the OS.
     */
    size_t trim_count() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));
        return trims_;
    }

    /**
     * @brief Check if the GC threads are running.
     */
    bool is_running() const {
        return running_.load(std::memory_order_relaxed);
    }
};
