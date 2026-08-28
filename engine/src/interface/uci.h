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

struct RootTreeResult {
    JointActionCandidate decision;
    NodeType rootType = NodeType::UNSOLVED;
    std::vector<RootEdgeStats> edges;
};

struct MergedRootResult {
    bool hasAction = false;
    JointActionCandidate action;
    NodeType rootType = NodeType::UNSOLVED;
    std::vector<RootEdgeStats> edges;
    size_t representativeTree = 0;
};

/** Merge independent per-GPU root trees by summing visits and visit-weighted Q. */
MergedRootResult merge_root_results(const std::vector<RootTreeResult>& trees);

class UCI {
private:
    std::thread* mainSearchThread;
    std::vector<std::unique_ptr<Agent>> agents;
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
    // Share one search graph across all CUDA engines by default. Independent
    // root trees remain available as an explicit fallback/benchmark mode.
    bool rootParallelism = false;
    // Remembered so agents created later still honour the configured size.
    size_t hashSizeMB = 0;
    SearchParams::RuntimeConfig searchConfig;

    // Rebuilds the engines (and the agent) with the current settings. The batch
    // size is baked into the TensorRT engine, so changing it means reloading.
    bool reload_engines();
    // Sizes the agent pool to the number of root trees the current mode needs.
    // Sharing one graph uses a single agent however many GPUs are present, and
    // each surplus agent would otherwise hold its own transposition table, GC
    // thread and idle worker pool.
    void ensure_agents(size_t count);
    size_t required_agent_count() const;
    JointActionCandidate run_root_parallel_search(const SearchOptions& options);
    void run_root_parallel_brain(const JointActionCandidate& played,
                                 const SearchOptions& options);

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
