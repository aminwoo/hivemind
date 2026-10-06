#pragma once

#include <atomic>
#include <string>
#include <sstream>
#include <thread>
#include <vector>
#include <memory>
#include <random>

#include "environment/board.h"
#include "environment/constants.h"
#include "search/searchthread.h"
#include "search/agent.h"
#include "nn/engine.h"
#include "nnue/network.h"
#include "search/alphabeta.h"

class UCI {
private:
    friend class UCIOpeningNoiseTestPeer;

    std::thread* mainSearchThread;
    std::unique_ptr<Agent> agent;
    Board board;
    Stockfish::Color teamSide = Stockfish::WHITE;
    // True when our team is ahead on the clocks, which is what makes sitting
    // and double-sitting legal. Set by the TimeAdvantage option.
    bool teamHasTimeAdvantage = false;
    std::vector<std::unique_ptr<Engine>> engines;
    // Retained so the BatchSize option can rebuild the engines in place.
    std::vector<int> deviceIds;
    std::string networkPath;
    int batchSize = SearchParams::BATCH_SIZE;
    std::atomic<bool> ongoingSearch{false};
    int multiPV = 1;  // Number of principal variations to display
    bool ponderEnabled = true;  // Whether to output ponder move and accept ponder search
    // Whether "go background" starts a search. Listed in the uci response so
    // a front end can tell this build accepts the command at all: an older
    // one would read it as a plain go and answer with a bestmove.
    bool backgroundSearchEnabled = true;
    SearchParams::RuntimeConfig searchConfig;
    // Optional early-game diversity for normal UCI play. A zero/noise-free
    // RuntimeConfig remains the default so existing users stay deterministic.
    bool openingNoiseEnabled = false;
    int openingNoisePlies = 16;
    int openingNoiseAlphaPermille = 100;
    int openingNoiseEpsilonPermille = 600;
    std::mt19937_64 openingNoiseGenerator;

    // Alpha-beta search on the distilled NNUE (SearchMode alphabeta). It needs
    // neither an ONNX model nor a GPU.
    bool alphaBetaMode = false;
    size_t hashMb = 16;
    std::unique_ptr<nnue::Network> nnueNetwork;
    std::unique_ptr<ab::Searcher> abSearcher;
    ab::Options abOptions;
    std::atomic<bool> abStop{false};
    std::atomic<bool> singleStop{false};
    std::atomic<bool> singlePonder{false};

    void go_single_board(int moveTime, size_t nodes, int depth, bool infinite, bool ponder);

    void go_alphabeta(int moveTime, size_t nodes, int depth, bool infinite);

    // Rebuilds the engines (and the agent) with the current settings. The batch
    // size is baked into the TensorRT engine, so changing it means reloading.
    bool reload_engines();

    // Returns the per-search config, enabling root noise only while both
    // boards are still inside the configured opening horizon.
    SearchParams::RuntimeConfig current_search_config();

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

    /// Loads an NNUE network and switches to alpha-beta search.
    bool load_nnue(const std::string& path);

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
