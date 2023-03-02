#ifndef UCI_H
#define UCI_H

#include <iostream> 
#include <thread>   

#include "bugboard.h"
#include "constants.h"
#include "searchthread.h"
#include "agent.h"

class UCI {
    private:
        std::thread* mainSearchThread; 
        Agent agent;
        Bugboard board;
        bool ongoingSearch = false;

    public:
        void go(std::istringstream& is);
        void stop();
        void position(std::istringstream& is);
        void time(std::istringstream& is);
        void loop();
};

#endif