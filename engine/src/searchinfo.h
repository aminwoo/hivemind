#pragma once

#include <chrono>
#include <algorithm>

struct SearchInfo {
    std::chrono::time_point<std::chrono::steady_clock> start;
    int moveTime; 
    int nodes = 0;
    int maxDepth = 0; 

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
        return nodes; 
    }

    inline int get_max_depth() const {
        return maxDepth; 
    }

    // Increments the node counter.
    inline void increment_nodes(int value) {
        nodes += value; 
    }

    // Updates the maximum depth encountered.
    inline void set_max_depth(int depth) {
        maxDepth = std::max(maxDepth, depth);
    }
};
