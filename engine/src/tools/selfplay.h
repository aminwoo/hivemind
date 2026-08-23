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
    float initialClockSeconds = 180.0f;
    int batchSize = SearchParams::BATCH_SIZE;
    uint64_t seed = 0;
    std::filesystem::path outputDirectory = "selfplay_games";
};

int run_selfplay(const std::vector<Engine*>& engines, const SelfPlayConfig& config);