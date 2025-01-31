#ifndef UCI_H
#define UCI_H

#include <iostream> 
#include <thread>   

#include "board.h"
#include "constants.h"
#include "searchthread.h"
#include "agent.h"

class UCI {
    private:
        std::thread* mainSearchThread; 
        Agent agent;
        Board board;
        Engine engine;
        bool ongoingSearch = false;

    public:
        void go(std::istringstream& is);
        void stop();
        void position(std::istringstream& is);
        void loop();
};

#endif
