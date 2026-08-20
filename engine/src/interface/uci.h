#pragma once

#include <atomic>
#include <string>
#include <sstream>
#include <thread>
#include <vector>
#include <memory>

#include "environment/board.h"
#include "environment/constants.h"
#include "search/searchthread.h"
#include "search/agent.h"
#include "nn/engine.h"

class UCI {
private:
    std::thread* mainSearchThread;
    std::unique_ptr<Agent> agent;
    Board board;
    Stockfish::Color teamSide = Stockfish::WHITE;
    bool teamHasTimeAdvantage = false;
    std::vector<std::unique_ptr<Engine>> engines;
    std::atomic<bool> ongoingSearch{false};
    int multiPV = 1;  // Number of principal variations to display
    bool ponderEnabled = true;  // Whether to output ponder move and accept ponder search
    SearchParams::RuntimeConfig searchConfig;

public:
    UCI();
    ~UCI();

    // Initialize engines on the specified GPU devices.
    // For each device ID in deviceIds, a new Engine is constructed.
    bool initializeEngines(
        const std::vector<int>& deviceIds,
        const std::string& networkPath = {});

    void send_uci_response();
    void go(std::istringstream& is);
    void ponderhit();
    void setoption(std::istringstream& is);
    void stop();
    void new_game();
    void position(std::istringstream& is);
    void policy();
    void loop();
};
