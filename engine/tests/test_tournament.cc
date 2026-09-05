#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>

#include "tools/tournament.h"

TEST(TournamentConfigTest, SelectsPwCoefficientByNetwork) {
    TournamentConfig config;
    config.contenderPwCoefficient = 1.5f;
    config.baselinePwCoefficient = 0.75f;
    config.contenderRootPwCoefficient = 1.5f;
    config.baselineRootPwCoefficient = 0.75f;

    EXPECT_FLOAT_EQ(config.pwCoefficientFor(true), 1.5f);
    EXPECT_FLOAT_EQ(config.pwCoefficientFor(false), 0.75f);
    EXPECT_FLOAT_EQ(config.searchConfigFor(true).pwCoefficient, 1.5f);
    EXPECT_FLOAT_EQ(config.searchConfigFor(true).rootPwCoefficient, 1.5f);
    EXPECT_FLOAT_EQ(config.searchConfigFor(false).pwCoefficient, 0.75f);
    EXPECT_FLOAT_EQ(config.searchConfigFor(false).rootPwCoefficient, 0.75f);
}

TEST(TournamentConfigTest, SelectsAllStrengthParametersByContestant) {
    TournamentConfig config;
    config.contenderMcgs = false;
    config.baselineRootMateSearch = false;
    config.contenderWdlWeight = 0.25f;
    config.baselineMovesLeftDiscount = 0.9f;
    config.contenderQValueWeight = 0.6f;
    config.baselineQVetoDelta = 0.15f;
    config.contenderSupplyPolicyWeight = 0.2f;
    config.contenderSupplyValueWeight = 0.3f;

    const auto contender = config.searchConfigFor(true);
    const auto baseline = config.searchConfigFor(false);
    EXPECT_FALSE(contender.enableMCGS);
    EXPECT_TRUE(baseline.enableMCGS);
    EXPECT_TRUE(contender.enableRootMateSearch);
    EXPECT_FALSE(baseline.enableRootMateSearch);
    EXPECT_FLOAT_EQ(contender.wdlValueWeight, 0.25f);
    EXPECT_FLOAT_EQ(baseline.movesLeftDiscount, 0.9f);
    EXPECT_FLOAT_EQ(contender.qValueWeight, 0.6f);
    EXPECT_FLOAT_EQ(baseline.qVetoDelta, 0.15f);
    EXPECT_FLOAT_EQ(contender.supplyPolicyWeight, 0.2f);
    EXPECT_FLOAT_EQ(contender.supplyValueWeight, 0.3f);
    EXPECT_FLOAT_EQ(baseline.supplyPolicyWeight, 0.0f);
    EXPECT_FLOAT_EQ(baseline.supplyValueWeight, 0.0f);
}

TEST(TournamentConfigTest, LoadsPairedRealPositions) {
    const auto path = std::filesystem::temp_directory_path()
        / "hivemind-tournament-positions.tsv";
    {
        std::ofstream stream(path);
        stream << "# dual fen, team, time advantage\n"
               << "8/8/8/8/8/8/8/K6k w - - 0 1|"
                  "8/8/8/8/8/8/8/K6k b - - 0 1\tblack\ttrue\n";
    }
    const auto positions = load_tournament_positions(path);
    std::filesystem::remove(path);
    ASSERT_EQ(positions.size(), 1U);
    EXPECT_EQ(positions[0].teamToPlay, Stockfish::BLACK);
    EXPECT_TRUE(positions[0].teamHasTimeAdvantage);
    EXPECT_NE(positions[0].dualFen.find('|'), std::string::npos);
}

TEST(TournamentResultTest, PairedSprtAcceptsEitherHypothesis) {
    std::vector<double> wins(200, 1.0);
    const SprtState strong = evaluate_paired_sprt(
        wins, 0.0, 10.0, 0.05, 0.05);
    EXPECT_EQ(strong.decision, SprtState::Decision::ACCEPT_H1);
    EXPECT_GE(strong.logLikelihoodRatio, strong.upperBoundary);

    std::vector<double> losses(200, 0.0);
    const SprtState weak = evaluate_paired_sprt(
        losses, 0.0, 10.0, 0.05, 0.05);
    EXPECT_EQ(weak.decision, SprtState::Decision::ACCEPT_H0);
    EXPECT_LE(weak.logLikelihoodRatio, weak.lowerBoundary);
}

TEST(TournamentResultTest, ReportsMeasuredFullSearchNps) {
    TournamentPerformance performance;
    performance.nodes = 2500;
    performance.nanoseconds = 500000000;
    EXPECT_DOUBLE_EQ(performance.nps(), 5000.0);
}

TEST(TournamentResultTest, ComputesScoreAndElo) {
    TournamentResult result;
    result.contenderWins = 6;
    result.baselineWins = 2;
    result.draws = 2;

    EXPECT_EQ(result.games(), 10);
    EXPECT_DOUBLE_EQ(result.contenderScore(), 0.7);
    ASSERT_TRUE(result.contenderElo().has_value());
    EXPECT_NEAR(*result.contenderElo(), 147.1907, 1e-3);
}

TEST(TournamentResultTest, EloIsUndefinedAtScoreEndpoints) {
    TournamentResult noGames;
    EXPECT_DOUBLE_EQ(noGames.contenderScore(), 0.0);
    EXPECT_FALSE(noGames.contenderElo().has_value());

    TournamentResult perfect;
    perfect.contenderWins = 2;
    EXPECT_FALSE(perfect.contenderElo().has_value());
}

TEST(TournamentResultTest, ComputesFirstTournamentConfidenceInterval) {
    TournamentResult result;
    result.contenderWins = 72;
    result.baselineWins = 28;
    result.pairScores.insert(result.pairScores.end(), 23, 1.0);
    result.pairScores.insert(result.pairScores.end(), 26, 0.5);
    result.pairScores.push_back(0.0);

    EXPECT_DOUBLE_EQ(result.contenderScore(), 0.72);
    ASSERT_TRUE(result.contenderElo().has_value());
    EXPECT_NEAR(*result.contenderElo(), 164.069786, 1e-6);

    const auto scoreInterval = result.scoreConfidence95();
    ASSERT_TRUE(scoreInterval.has_value());
    EXPECT_NEAR(scoreInterval->first, 0.645078483, 1e-9);
    EXPECT_NEAR(scoreInterval->second, 0.794921517, 1e-9);

    const auto eloInterval = result.eloConfidence95();
    ASSERT_TRUE(eloInterval.has_value());
    EXPECT_NEAR(eloInterval->first, 103.792091, 1e-6);
    EXPECT_NEAR(eloInterval->second, 235.361663, 1e-6);
    EXPECT_EQ(result.confidenceMethod(), "paired-opening normal approximation");
}

TEST(TournamentBreakdownTest, CountsAllOutcomes) {
    TournamentBreakdown breakdown;
    breakdown.wins = 3;
    breakdown.losses = 1;
    breakdown.draws = 2;
    EXPECT_EQ(breakdown.games(), 6);
}
