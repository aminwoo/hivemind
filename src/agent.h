#ifndef AGENT_H
#define AGENT_H

#include "searchthread.h"
#include "board.h"
#include "node.h"
#include "engine.h"

class Agent {
private:
    int numberOfThreads; 
    std::vector<SearchThread*> searchThreads; 
    std::mutex mtx;
    bool running;

public:
    Agent(int numThreads);
    ~Agent();

    void run_search(Board& board, const std::vector<Engine*>& engines, int move_time); 
    void set_is_running(bool value);  
    bool is_running(); 
};

#endif