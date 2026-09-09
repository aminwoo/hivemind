#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "search/node.h"

/**
 * @brief Garbage Collection Thread for async tree cleanup.
 * 
 * Frees old search tree subtrees asynchronously to avoid latency spikes
 * during time-critical search operations. When a search completes and
 * tree reuse is enabled, the unused portions of the old tree are queued
 * for async deletion by this thread.
 * 
 * Based on CrazyAra's GCThread implementation.
 */
class GCThread {
public:
    /**
     * @brief Trees allowed to sit unfreed before a producer has to wait.
     *
     * A search tree is hundreds of megabytes, so an unbounded queue means a
     * producer that outruns this thread keeps every tree it discarded resident
     * at once. Two - one draining, one waiting - lets the common case enqueue
     * without ever blocking while capping what a burst can hold.
     */
    static constexpr size_t kDefaultCapacity = 2;

    /// Least time between malloc_trim calls, which walk every arena.
    static constexpr std::chrono::seconds kTrimInterval{5};

private:
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable roomCv_;
    std::queue<std::shared_ptr<Node>> deleteQueue_;
    size_t capacity_{kDefaultCapacity};
    std::atomic<bool> running_{false};
    std::atomic<bool> terminate_{false};
    std::chrono::steady_clock::time_point lastTrim_{};

    /**
     * @brief Return the freed arenas to the OS, at most once per interval.
     *
     * Freeing a tree hands its chunks back to glibc, not to the kernel: with a
     * thread per search worker the per-thread arenas keep them, and RSS stays
     * at the high-water mark of the largest tree the process ever held. This
     * runs on the GC thread with nothing waiting on it, which is the one place
     * the walk costs nothing.
     */
    void trim_arenas() {
#if defined(__GLIBC__)
        const auto now = std::chrono::steady_clock::now();
        if (now - lastTrim_ < kTrimInterval) {
            return;
        }
        lastTrim_ = now;
        malloc_trim(0);
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
            std::shared_ptr<Node> nodeToDelete;
            bool drained = false;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] {
                    return !deleteQueue_.empty() || terminate_.load(std::memory_order_relaxed);
                });

                if (terminate_.load(std::memory_order_relaxed) && deleteQueue_.empty()) {
                    break;
                }

                if (!deleteQueue_.empty()) {
                    nodeToDelete = std::move(deleteQueue_.front());
                    deleteQueue_.pop();
                    drained = deleteQueue_.empty();
                }
            }

            // Release the node outside the lock (actual deletion happens here)
            nodeToDelete.reset();
            // A producer waiting for room can proceed as soon as the slot is
            // free, which is now rather than after the next wait.
            roomCv_.notify_all();
            if (drained) {
                trim_arenas();
            }
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
     * @brief Start the garbage collection thread.
     */
    void start() {
        if (running_.load(std::memory_order_relaxed)) return;
        
        terminate_.store(false, std::memory_order_relaxed);
        running_.store(true, std::memory_order_relaxed);
        worker_ = std::thread(&GCThread::worker_loop, this);
    }
    
    /**
     * @brief Stop the garbage collection thread.
     * Waits for all pending deletions to complete.
     */
    void stop() {
        if (!running_.load(std::memory_order_relaxed)) return;
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            terminate_.store(true, std::memory_order_relaxed);
        }
        cv_.notify_one();
        // Release any producer parked for a slot; it frees its own tree once
        // it sees the termination flag.
        roomCv_.notify_all();

        if (worker_.joinable()) {
            worker_.join();
        }

        running_.store(false, std::memory_order_relaxed);
    }
    
    /**
     * @brief Queue a node subtree for async deletion.
     * 
     * The node and all its children will be deleted asynchronously.
     * @param node The root of the subtree to delete
     */
    void enqueue(std::shared_ptr<Node> node) {
        if (!node) return;

        // Freeing off-thread hides the latency; it must not also hide how much
        // is still waiting to be freed. If this thread has fallen behind, the
        // producer waits for a slot rather than letting another whole tree
        // stay resident. When the GC keeps up - the ordinary case - the
        // predicate already holds and nothing blocks.
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (running_.load(std::memory_order_relaxed)) {
                roomCv_.wait(lock, [this] {
                    return deleteQueue_.size() < capacity_
                        || terminate_.load(std::memory_order_relaxed);
                });
            }
            if (terminate_.load(std::memory_order_relaxed)
                || !running_.load(std::memory_order_relaxed)) {
                // Nothing will drain the queue. Free it here instead of
                // parking a tree that would outlive the thread meant to
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
     * @brief Set how many trees may wait unfreed before enqueue() blocks.
     * @param capacity Queue depth; clamped to at least one.
     */
    void set_capacity(size_t capacity) {
        std::lock_guard<std::mutex> lock(mutex_);
        capacity_ = std::max<size_t>(1, capacity);
        roomCv_.notify_all();
    }
    
    /**
     * @brief Get the number of pending deletions.
     */
    size_t pending_count() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));
        return deleteQueue_.size();
    }
    
    /**
     * @brief Check if the GC thread is running.
     */
    bool is_running() const {
        return running_.load(std::memory_order_relaxed);
    }
};
