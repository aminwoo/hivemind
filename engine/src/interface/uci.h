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
    // Retained so the BatchSize option can rebuild the engines in place.
    std::vector<int> deviceIds;
    std::string networkPath;
    int batchSize = SearchParams::BATCH_SIZE;
    std::atomic<bool> ongoingSearch{false};
    int multiPV = 1;  // Number of principal variations to display
    bool ponderEnabled = true;  // Whether to output ponder move and accept ponder search
    SearchParams::RuntimeConfig searchConfig;

    // Rebuilds the engines (and the agent) with the current settings. The batch
    // size is baked into the TensorRT engine, so changing it means reloading.
    bool reload_engines();

public:
    UCI();
    ~UCI();

    // Initialize engines on the specified GPU devices.
    // For each device ID in deviceIds, a new Engine is constructed.
    // The arguments are retained so reload_engines() can repeat the setup.
    bool initializeEngines(
        const std::vector<int>& deviceIdsToUse,
        const std::string& networkPathToUse = {},
        int batchSizeToUse = SearchParams::BATCH_SIZE);

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
