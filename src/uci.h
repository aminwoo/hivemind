#pragma once

#include <thread>
#include <sstream>
#include <vector>
#include <memory>

#include "board.h"
#include "constants.h"
#include "searchthread.h"
#include "agent.h"
#include "engine.h"

class UCI {
private:
    std::thread* mainSearchThread;
    std::unique_ptr<Agent> agent;
    Board board;
    std::vector<std::unique_ptr<Engine>> engines;
    bool ongoingSearch = false;

public:
    UCI();
    ~UCI();

    // Initialize engines on the specified GPU devices.
    // For each device ID in deviceIds, a new Engine is constructed.
    void initializeEngines(const std::vector<int>& deviceIds);

    void go(std::istringstream& is);
    void stop();
    void position(std::istringstream& is);
    void loop();
};
