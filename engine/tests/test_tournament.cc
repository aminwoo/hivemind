#include <gtest/gtest.h>

#include <cmath>

#include "tournament.h"

TEST(TournamentConfigTest, SelectsPassPriorFloorsByNetwork) {
    TournamentConfig config;
    config.contenderPassPriorFloors = {0.0f, 0.0f};
    config.baselinePassPriorFloors = {0.10f, 0.05f};

    EXPECT_FLOAT_EQ(config.passPriorFloorsFor(true).wait, 0.0f);
    EXPECT_FLOAT_EQ(config.passPriorFloorsFor(true).coordination, 0.0f);
    EXPECT_FLOAT_EQ(config.passPriorFloorsFor(false).wait, 0.10f);
    EXPECT_FLOAT_EQ(config.passPriorFloorsFor(false).coordination, 0.05f);
}

TEST(TournamentConfigTest, SelectsPwCoefficientByNetwork) {
    TournamentConfig config;
    config.contenderPwCoefficient = 1.5f;
    config.baselinePwCoefficient = 0.75f;

    EXPECT_FLOAT_EQ(config.pwCoefficientFor(true), 1.5f);
    EXPECT_FLOAT_EQ(config.pwCoefficientFor(false), 0.75f);
    EXPECT_FLOAT_EQ(config.searchConfigFor(true).pwCoefficient, 1.5f);
    EXPECT_FLOAT_EQ(config.searchConfigFor(true).rootPwCoefficient, 1.5f);
    EXPECT_FLOAT_EQ(config.searchConfigFor(false).pwCoefficient, 0.75f);
    EXPECT_FLOAT_EQ(config.searchConfigFor(false).rootPwCoefficient, 0.75f);
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