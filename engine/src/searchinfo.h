#pragma once

#include <atomic>
#include <chrono>
#include <algorithm>
#include <mutex>

/**
 * @brief Thread-safe search statistics tracking.
 * 
 * Uses atomic operations for counters to support multi-threaded MCTS.
 * Includes time management state for early stopping and time extension.
 */
struct SearchInfo {
    std::chrono::time_point<std::chrono::steady_clock> start;
    int moveTime; 
    std::atomic<int> nodes{0};
    std::atomic<int> maxDepth{0};
    std::atomic<int> collisions{0};
    
    // Time management state
    mutable std::mutex timeMutex_;
    float lastRootEval_ = 0.0f;        // Last recorded root evaluation
    float overallNPS_ = 0.0f;          // Running average of nodes per second
    int timeExtensionCount_ = 0;       // Number of time extensions applied
    int effectiveMoveTime_ = 0;        // Current move time (may be extended)
    bool inGame_ = false;              // Whether this is a timed game

    // Constructor initializes the start time and move time.
    SearchInfo(std::chrono::time_point<std::chrono::steady_clock> start, int moveTime)
        : start(start), moveTime(moveTime), effectiveMoveTime_(moveTime) {}
    
    ~SearchInfo() = default;

    // Returns the original move time.
    int get_move_time() const {
        return moveTime; 
    }
    
    // Returns the effective move time (may be extended).
    int get_effective_move_time() const {
        std::lock_guard<std::mutex> lock(timeMutex_);
        return effectiveMoveTime_;
    }
    
    // Returns remaining time in milliseconds.
    double remaining_time() const {
        std::lock_guard<std::mutex> lock(timeMutex_);
        return std::max(0.0, effectiveMoveTime_ - elapsed_unlocked());
    }

    // Returns the elapsed time in milliseconds since start (no lock).
    double elapsed_unlocked() const {
        auto elapsed_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        return elapsed_duration.count(); 
    }

    // Returns the elapsed time in milliseconds since start.
    double elapsed() const {
        return elapsed_unlocked();
    }

    inline int get_nodes_searched() const {
        return nodes.load(std::memory_order_relaxed); 
    }

    inline int get_max_depth() const {
        return maxDepth.load(std::memory_order_relaxed); 
    }

    // Atomically increments the node counter.
    inline void increment_nodes(int value) {
        nodes.fetch_add(value, std::memory_order_relaxed); 
    }

    // Atomically updates the maximum depth encountered.
    inline void set_max_depth(int depth) {
        int current = maxDepth.load(std::memory_order_relaxed);
        while (depth > current && 
               !maxDepth.compare_exchange_weak(current, depth, std::memory_order_relaxed));
    }

    // Atomically increments the collision counter.
    inline void increment_collisions(int value = 1) {
        collisions.fetch_add(value, std::memory_order_relaxed);
    }

    // Returns the collision count.
    inline int get_collisions() const {
        return collisions.load(std::memory_order_relaxed);
    }
    
    // =========================================================================
    // Time Management Methods
    // =========================================================================
    
    /**
     * @brief Update the running NPS average.
     * Called periodically during search.
     */
    void update_nps() {
        std::lock_guard<std::mutex> lock(timeMutex_);
        double elapsedSec = elapsed_unlocked() / 1000.0;
        if (elapsedSec > 0.1) {  // Need at least 100ms for reliable NPS
            float currentNPS = static_cast<float>(nodes.load(std::memory_order_relaxed) / elapsedSec);
            if (overallNPS_ == 0.0f) {
                overallNPS_ = currentNPS;
            } else {
                // Exponential moving average
                overallNPS_ = 0.9f * overallNPS_ + 0.1f * currentNPS;
            }
        }
    }
    
    /**
     * @brief Get the current NPS estimate.
     */
    float get_nps() const {
        std::lock_guard<std::mutex> lock(timeMutex_);
        return overallNPS_;
    }
    
    /**
     * @brief Set whether this is an in-game search with time controls.
     */
    void set_in_game(bool inGame) {
        std::lock_guard<std::mutex> lock(timeMutex_);
        inGame_ = inGame;
    }
    
    /**
     * @brief Check if this is an in-game search.
     */
    bool is_in_game() const {
        std::lock_guard<std::mutex> lock(timeMutex_);
        return inGame_;
    }
    
    /**
     * @brief Update the last root evaluation.
     * @param eval Current root Q-value
     */
    void set_last_eval(float eval) {
        std::lock_guard<std::mutex> lock(timeMutex_);
        lastRootEval_ = eval;
    }
    
    /**
     * @brief Get the last root evaluation.
     */
    float get_last_eval() const {
        std::lock_guard<std::mutex> lock(timeMutex_);
        return lastRootEval_;
    }
    
    /**
     * @brief Try to extend the move time.
     * @param factor Multiplication factor for remaining time
     * @param maxExtensions Maximum number of extensions allowed
     * @return true if extension was applied, false if limit reached
     */
    bool try_extend_time(float factor, int maxExtensions) {
        std::lock_guard<std::mutex> lock(timeMutex_);
        if (timeExtensionCount_ >= maxExtensions) {
            return false;
        }
        
        double remaining = effectiveMoveTime_ - elapsed_unlocked();
        if (remaining <= 0) {
            return false;
        }
        
        int extension = static_cast<int>(remaining * (factor - 1.0f));
        effectiveMoveTime_ += extension;
        timeExtensionCount_++;
        return true;
    }
    
    /**
     * @brief Get the number of time extensions applied.
     */
    int get_extension_count() const {
        std::lock_guard<std::mutex> lock(timeMutex_);
        return timeExtensionCount_;
    }
};
