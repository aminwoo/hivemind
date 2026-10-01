#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "search/search_params.h"

class Engine;

struct SelfPlayConfig {
    size_t games = 1;
    size_t nodes = 800;
    size_t maxMacroPlies = 400;
    size_t chunkSamples = 16384;
    double rawPolicyMeanMacroPlies = 8.0;
    size_t rawPolicyMaxMacroPlies = 30;
    double rawPolicyHighTemperatureProbability = 0.05;
    double mctsTemperature = 1.0;
    double mctsTemperatureDecay = 0.93;
    size_t mctsTemperaturePlies = 20;
    float resignThreshold = -0.90f;
    size_t resignConsecutivePlies = 3;
    double resignDisableFraction = 0.10;
    double qValueRatio = 0.15;
    double nodeRandomFactor = 0.05;
    float dirichletAlpha = 0.3f;
    float dirichletEpsilon = 0.25f;
    uint64_t fairyStockfishMateNodes = SearchParams::MATE_PROBE_ROOT_NODE_BUDGET;
    float initialClockSeconds = 180.0f;
    int batchSize = SearchParams::BATCH_SIZE;
    uint64_t seed = 0;
    std::filesystem::path outputDirectory = "selfplay_games";
    // When set, every searched position is also written as an HDST
    // distillation record (tools/distill_data.h): the visit distribution over
    // all legal moves of each board on turn, root Q as the value, and for
    // finished games the result as WDL plus remaining team decisions.
    // Games played at once, each on its own engine (see main: one engine per
    // game on each device). 1 keeps a single game spread over all engines.
    size_t parallelGames = 1;
    // Write the HVM5 training chunks (off when only distillation data is wanted).
    bool writeTrainingChunks = true;
    // Root mate scan and selected-move certification before each search.
    bool rootScans = true;
    std::filesystem::path distillOutputDirectory;
    size_t distillChunkPositions = 250'000;
};

// Game-indexed streams stay independent of worker assignment.
inline uint64_t selfplay_seed(uint64_t seed, uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return seed ^ (value ^ (value >> 31));
}

int run_selfplay(const std::vector<Engine*>& engines, const SelfPlayConfig& config);