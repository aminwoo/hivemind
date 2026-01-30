#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "node.h"

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
private:
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::shared_ptr<Node>> deleteQueue_;
    std::atomic<bool> running_{false};
    std::atomic<bool> terminate_{false};
    
    /**
     * @brief Worker thread loop that processes delete requests.
     */
    void worker_loop() {
        while (!terminate_.load(std::memory_order_relaxed)) {
            std::shared_ptr<Node> nodeToDelete;
            
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
                }
            }
            
            // Release the node outside the lock (actual deletion happens here)
            nodeToDelete.reset();
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
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            deleteQueue_.push(std::move(node));
        }
        cv_.notify_one();
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
