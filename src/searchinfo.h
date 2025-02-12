#pragma once

#include <mutex>
#include <chrono>
#include <algorithm>

struct SearchInfo {
    std::mutex mtx;
    std::chrono::time_point<std::chrono::steady_clock> start;
    int move_time; 
    int nodes = 0;
    int maxDepth = 0;  
    int collisions = 0; 

    // Constructor initializes the start time and move time.
    SearchInfo(std::chrono::time_point<std::chrono::steady_clock> start, int move_time)
        : start(start), move_time(move_time) {}
    
    ~SearchInfo() = default;

    // Returns the move time.
    int get_move_time() const {
        return move_time; 
    }

    // Returns the elapsed time in milliseconds since start.
    double elapsed() const {
        auto elapsed_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        return elapsed_duration.count(); 
    }

    inline int get_nodes_searched() const {
        return nodes; 
    }

    inline int get_max_depth() const {
        return maxDepth; 
    }

    inline int get_collisions() const {
        return collisions; 
    }

    // Safely increments the node counter.
    inline void increment_nodes(int value) {
        std::lock_guard<std::mutex> lock(mtx);
        nodes += value; 
    }

    // Safely increments the collisions counter.
    inline void increment_collisions(int value) {
        std::lock_guard<std::mutex> lock(mtx);
        collisions += value; 
    }

    // Safely updates the maximum depth encountered.
    inline void set_max_depth(int depth) {
        std::lock_guard<std::mutex> lock(mtx);
        maxDepth = std::max(maxDepth, depth);
    }
};
