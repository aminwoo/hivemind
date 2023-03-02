#ifndef AGENT_H
#define AGENT_H

#include <iomanip>

#include "searchthread.h"
#include "bugboard.h"
#include "node.h"

class Agent {
    private:
        int numberOfThreads = 1; 
        std::vector<SearchThread*> searchThreads; 
        std::mutex mtx;
        bool running = false; 
    public:
        Agent();
        ~Agent();
        void run_search(Bugboard& board, int move_time); 
        void set_is_running(bool value);  
        bool is_running(); 
};

#endif