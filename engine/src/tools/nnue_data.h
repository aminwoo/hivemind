#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

class Engine;

/**
 * @brief Generates NNUE distillation data from teacher-policy games.
 *
 * Games are played by sampling the teacher network's raw policy, with a
 * per-game temperature and a small chance of a uniformly random move so the
 * student also sees the positions a search will reach after poor moves. Every
 * non-terminal position is written with its sparse NNUE features (side to
 * move only), the teacher's value and WDL probabilities, and the game result.
 */
struct NnueDataConfig {
    uint64_t positions = 1'000'000;
    size_t threads = 4;
    size_t maxMacroPlies = 300;
    size_t chunkPositions = 1'000'000;
    double randomMoveProbability = 0.03;
    bool writeFens = false;
    uint64_t seed = 0;
    std::filesystem::path outputDirectory = "nnue_data";
};

int run_nnue_data(Engine& engine, const NnueDataConfig& config);
