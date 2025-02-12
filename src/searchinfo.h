#ifndef SEARCHINFO_H
#define SEARCHINFO_H

#include <mutex>

struct SearchInfo {
    std::mutex mtx;
    std::chrono::time_point<std::chrono::steady_clock>  start;
    int move_time; 
    int nodes = 0;
    int maxDepth = 0;  
    int collisions = 0; 

    SearchInfo(std::chrono::time_point<std::chrono::steady_clock>  start, int move_time) : start(start), move_time(move_time) {};
    ~SearchInfo() {}; 

    int get_move_time() {
        return move_time; 
    }

    double elapsed() {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        return elapsed.count(); 
    }

    inline int get_nodes_searched() {
        return nodes; 
    }

    inline int get_max_depth() {
        return maxDepth; 
    }

    inline int get_collisions() {
        return collisions; 
    }

    inline void increment_nodes(int value) {
        mtx.lock();
        nodes += value; 
        mtx.unlock(); 
    }

    inline void increment_colllisions(int value) {
        mtx.lock();
        collisions += value; 
        mtx.unlock(); 
    }

    inline void set_max_depth(int depth) {
        mtx.lock();
        maxDepth = std::max(maxDepth, depth);
        mtx.unlock(); 
    }

};

#endif
