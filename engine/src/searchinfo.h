#pragma once

#include <atomic>
#include <chrono>
#include <algorithm>

/**
 * @brief Thread-safe search statistics tracking.
 * 
 * Uses atomic operations for counters to support multi-threaded MCTS.
 */
struct SearchInfo {
    std::chrono::time_point<std::chrono::steady_clock> start;
    int moveTime; 
    std::atomic<int> nodes{0};
    std::atomic<int> maxDepth{0};
    std::atomic<int> collisions{0};

    // Constructor initializes the start time and move time.
    SearchInfo(std::chrono::time_point<std::chrono::steady_clock> start, int moveTime)
        : start(start), moveTime(moveTime) {}
    
    ~SearchInfo() = default;

    // Returns the move time.
    int get_move_time() const {
        return moveTime; 
    }

    // Returns the elapsed time in milliseconds since start.
    double elapsed() const {
        auto elapsed_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        return elapsed_duration.count(); 
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
};
