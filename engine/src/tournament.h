#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

class Engine;

struct TournamentConfig {
    size_t games = 20;
    size_t nodes = 400;
    size_t maxMacroPlies = 400;
    float dirichletAlpha = 0.3f;
    float dirichletEpsilon = 0.10f;
    uint64_t seed = 1;
    std::filesystem::path outputDirectory = "tournament_results";
};

struct TournamentBreakdown {
    size_t wins = 0;
    size_t losses = 0;
    size_t draws = 0;

    size_t games() const;
};

struct TournamentResult {
    size_t contenderWins = 0;
    size_t baselineWins = 0;
    size_t draws = 0;
    TournamentBreakdown asWhite;
    TournamentBreakdown asBlack;
    TournamentBreakdown upTime;
    TournamentBreakdown downTime;
    size_t checkmates = 0;
    size_t noLegalActions = 0;
    size_t drawnTerminations = 0;
    size_t macroPlyLimits = 0;
    std::vector<double> pairScores;

    size_t games() const;
    double contenderScore() const;
    std::optional<double> contenderElo() const;
    std::optional<std::pair<double, double>> scoreConfidence95() const;
    std::optional<std::pair<double, double>> eloConfidence95() const;
    std::string confidenceMethod() const;
};

int run_tournament(
    Engine& contender,
    Engine& baseline,
    const std::string& contenderName,
    const std::string& baselineName,
    const TournamentConfig& config);