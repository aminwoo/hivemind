#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

class Engine;

struct SelfPlayConfig {
    size_t games = 1;
    size_t nodes = 400;
    size_t maxMacroPlies = 400;
    size_t chunkSamples = 16384;
    double rawPolicyMeanMacroPlies = 8.0;
    size_t rawPolicyMaxMacroPlies = 30;
    double rawPolicyHighTemperatureProbability = 0.05;
    double mctsTemperature = 0.8;
    double mctsTemperatureDecay = 0.93;
    double nodeRandomFactor = 0.05;
    float dirichletAlpha = 0.3f;
    float dirichletEpsilon = 0.25f;
    float initialClockSeconds = 180.0f;
    uint64_t seed = 0;
    std::filesystem::path outputDirectory = "selfplay_games";
};

int run_selfplay(Engine& engine, const SelfPlayConfig& config);