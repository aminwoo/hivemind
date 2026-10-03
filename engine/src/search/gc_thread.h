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

private:
    std::vector<std::thread> workers_;
    size_t workerCount_{kDefaultWorkers};
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable roomCv_;
    std::queue<std::shared_ptr<void>> deleteQueue_;
    size_t capacity_{kDefaultCapacity};
    size_t freeing_{0};  // Items taken off the queue and still being freed.
    std::atomic<bool> running_{false};
    std::atomic<bool> terminate_{false};
    std::chrono::steady_clock::time_point lastTrim_{};

    /**
     * @brief Return the freed arenas to the OS, at most once per interval.
     *
     * Freeing a tree hands its chunks back to glibc, not to the kernel: with a
     * thread per search worker the per-thread arenas keep them, and RSS stays
     * at the high-water mark of the largest tree the process ever held. This
     * runs on a GC thread with nothing waiting on it, which is the one place
     * the walk costs nothing. The caller holds mutex_, so only one thread at
     * a time decides to trim.
     */
    bool should_trim_locked() {
#if defined(__GLIBC__)
        const auto now = std::chrono::steady_clock::now();
        if (now - lastTrim_ < kTrimInterval) {
            return false;
        }
        lastTrim_ = now;
        return true;
#else
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

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] {
                    return !deleteQueue_.empty() || terminate_.load(std::memory_order_relaxed);
                });

                if (terminate_.load(std::memory_order_relaxed) && deleteQueue_.empty()) {
                    break;
                }

                nodeToDelete = std::move(deleteQueue_.front());
                deleteQueue_.pop();
                ++freeing_;
            }
            // A producer waiting for room can proceed as soon as the slot is
            // free, which is now rather than after this item is freed.
            roomCv_.notify_all();

            // Release the node outside the lock (actual deletion happens here)
            nodeToDelete.reset();

            bool trim = false;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --freeing_;
                // Trim once everything handed over is gone, not between items
                // another thread is still freeing.
                trim = deleteQueue_.empty() && freeing_ == 0
                    && should_trim_locked();
            }
#if defined(__GLIBC__)
            if (trim) {
                malloc_trim(0);
            }
#endif
        }
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
     * @brief Check if the GC threads are running.
     */
    bool is_running() const {
        return running_.load(std::memory_order_relaxed);
    }
};
