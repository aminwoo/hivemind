#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "search/search_params.h"
#include "Fairy-Stockfish/src/types.h"

class Engine;

struct TournamentConfig {
    size_t games = 20;
    size_t nodes = 400;
    int moveTimeMs = 0;
    int contenderBatchSize = SearchParams::BATCH_SIZE;
    int baselineBatchSize = SearchParams::BATCH_SIZE;
    int contenderThreads = SearchParams::NUM_SEARCH_THREADS;
    int baselineThreads = SearchParams::NUM_SEARCH_THREADS;
    size_t maxMacroPlies = 400;
    float dirichletAlpha = 0.3f;
    float dirichletEpsilon = 0.10f;
    float contenderPwCoefficient = SearchParams::PW_COEFFICIENT;
    float baselinePwCoefficient = SearchParams::PW_COEFFICIENT;
    float contenderRootPwCoefficient = SearchParams::ROOT_PW_COEFFICIENT;
    float baselineRootPwCoefficient = SearchParams::ROOT_PW_COEFFICIENT;
    bool contenderMcgs = SearchParams::ENABLE_MCGS;
    bool baselineMcgs = SearchParams::ENABLE_MCGS;
    bool contenderTranspositions = SearchParams::ENABLE_TRANSPOSITIONS;
    bool baselineTranspositions = SearchParams::ENABLE_TRANSPOSITIONS;
    bool contenderRootMateSearch = SearchParams::ENABLE_MATE_EARLY_EXIT;
    bool baselineRootMateSearch = SearchParams::ENABLE_MATE_EARLY_EXIT;
    bool contenderWdlEval = SearchParams::ENABLE_WDL_EVAL;
    bool baselineWdlEval = SearchParams::ENABLE_WDL_EVAL;
    float contenderWdlWeight = SearchParams::WDL_VALUE_WEIGHT;
    float baselineWdlWeight = SearchParams::WDL_VALUE_WEIGHT;
    float contenderMovesLeftDiscount = SearchParams::MOVES_LEFT_DISCOUNT;
    float baselineMovesLeftDiscount = SearchParams::MOVES_LEFT_DISCOUNT;
    float contenderQValueWeight = SearchParams::Q_VALUE_WEIGHT;
    float baselineQValueWeight = SearchParams::Q_VALUE_WEIGHT;
    float contenderQVetoDelta = SearchParams::Q_VETO_DELTA;
    float baselineQVetoDelta = SearchParams::Q_VETO_DELTA;
    std::filesystem::path positionsFile;
    std::string contenderModelSignature;
    std::string baselineModelSignature;
    double sprtElo0 = 0.0;
    double sprtElo1 = 0.0;
    double sprtAlpha = 0.05;
    double sprtBeta = 0.05;
    uint64_t seed = 1;
    std::filesystem::path outputDirectory = "tournament_results";

    float pwCoefficientFor(bool isContender) const {
        return isContender ? contenderPwCoefficient : baselinePwCoefficient;
    }

    SearchParams::RuntimeConfig searchConfigFor(bool isContender) const {
        SearchParams::RuntimeConfig searchConfig;
        searchConfig.pwCoefficient = pwCoefficientFor(isContender);
        searchConfig.rootPwCoefficient = isContender
            ? contenderRootPwCoefficient : baselineRootPwCoefficient;
        searchConfig.enableMCGS = isContender ? contenderMcgs : baselineMcgs;
        searchConfig.enableTranspositions = isContender
            ? contenderTranspositions : baselineTranspositions;
        searchConfig.enableRootMateSearch = isContender
            ? contenderRootMateSearch : baselineRootMateSearch;
        searchConfig.enableWdlEval = isContender
            ? contenderWdlEval : baselineWdlEval;
        searchConfig.wdlValueWeight = isContender
            ? contenderWdlWeight : baselineWdlWeight;
        searchConfig.movesLeftDiscount = isContender
            ? contenderMovesLeftDiscount : baselineMovesLeftDiscount;
        searchConfig.qValueWeight = isContender
            ? contenderQValueWeight : baselineQValueWeight;
        searchConfig.qVetoDelta = isContender
            ? contenderQVetoDelta : baselineQVetoDelta;
        return searchConfig;
    }

    bool sprtEnabled() const { return sprtElo1 > sprtElo0; }
};

struct TournamentStartPosition {
    std::string dualFen;
    Stockfish::Color teamToPlay = Stockfish::WHITE;
    bool teamHasTimeAdvantage = false;
};

std::vector<TournamentStartPosition> load_tournament_positions(
    const std::filesystem::path& path);

struct SprtState {
    enum class Decision { CONTINUE, ACCEPT_H0, ACCEPT_H1 };
    Decision decision = Decision::CONTINUE;
    double logLikelihoodRatio = 0.0;
    double lowerBoundary = 0.0;
    double upperBoundary = 0.0;
};

SprtState evaluate_paired_sprt(const std::vector<double>& pairScores,
                               double elo0, double elo1,
                               double alpha, double beta);

struct TournamentPerformance {
    uint64_t searches = 0;
    uint64_t nodes = 0;
    uint64_t nanoseconds = 0;

    double nps() const {
        return nanoseconds == 0 ? 0.0
            : 1.0e9 * static_cast<double>(nodes)
                / static_cast<double>(nanoseconds);
    }
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
    SprtState sprt;
    TournamentPerformance contenderPerformance;
    TournamentPerformance baselinePerformance;

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
