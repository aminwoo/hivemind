#include <gtest/gtest.h>

#include <cstdlib>
#include <random>
#include <stdexcept>

#include "tools/clock_sampling.h"

TEST(ClockSamplingTest, LogUniformStaysInBoundsAndCentresOnTheGeometricMean) {
    std::mt19937_64 random(7);
    size_t belowGeometricMean = 0;
    constexpr size_t draws = 4000;
    for (size_t draw = 0; draw < draws; ++draw) {
        const int32_t value = sample_log_uniform(100, 10'000, random);
        ASSERT_GE(value, 100);
        ASSERT_LE(value, 10'000);
        belowGeometricMean += value < 1'000;
    }
    EXPECT_NEAR(static_cast<double>(belowGeometricMean) / draws, 0.5, 0.05);
}

TEST(ClockSamplingTest, EqualBoundsGiveThatValueWithoutDrawing) {
    std::mt19937_64 random(1);
    std::mt19937_64 untouched(1);
    EXPECT_EQ(sample_log_uniform(1'000, 1'000, random), 1'000);
    EXPECT_EQ(random(), untouched());
}

TEST(ClockSamplingTest, LevelFractionControlsLevelStarts) {
    std::mt19937_64 random(3);
    ClockSampling alwaysLevel;
    alwaysLevel.levelStartFraction = 1.0;
    for (int draw = 0; draw < 100; ++draw) {
        EXPECT_EQ(sample_start_lead(alwaysLevel, random), 0);
    }

    ClockSampling neverLevel;
    neverLevel.levelStartFraction = 0.0;
    bool sawAhead = false;
    bool sawBehind = false;
    for (int draw = 0; draw < 200; ++draw) {
        const int32_t lead = sample_start_lead(neverLevel, random);
        ASSERT_NE(lead, 0);
        ASSERT_GE(std::abs(lead), neverLevel.startLeadMinMs);
        ASSERT_LE(std::abs(lead), neverLevel.startLeadMaxMs);
        sawAhead |= lead > 0;
        sawBehind |= lead < 0;
    }
    EXPECT_TRUE(sawAhead);
    EXPECT_TRUE(sawBehind);
}

TEST(ClockSamplingTest, DefaultHalfOfGamesStartLevel) {
    std::mt19937_64 random(11);
    const ClockSampling sampling;
    size_t level = 0;
    constexpr size_t draws = 4000;
    for (size_t draw = 0; draw < draws; ++draw) {
        level += sample_start_lead(sampling, random) == 0;
    }
    EXPECT_NEAR(static_cast<double>(level) / draws, 0.5, 0.05);
}

TEST(ClockSamplingTest, ClockDrawsEachTeamsMoveTimeSeparately) {
    std::mt19937_64 random(5);
    const ClockSampling sampling;
    bool sawDifferentMoveTimes = false;
    for (int draw = 0; draw < 50; ++draw) {
        const TimeState clock = sample_clock(sampling, random);
        ASSERT_TRUE(clock.is_clocked());
        EXPECT_EQ(clock.lead_ms(Stockfish::WHITE), -clock.lead_ms(Stockfish::BLACK));
        for (const Stockfish::Color team : {Stockfish::WHITE, Stockfish::BLACK}) {
            EXPECT_GE(clock.move_time_ms(team), sampling.moveTimeMinMs);
            EXPECT_LE(clock.move_time_ms(team), sampling.moveTimeMaxMs);
        }
        sawDifferentMoveTimes |= clock.move_time_ms(Stockfish::WHITE)
            != clock.move_time_ms(Stockfish::BLACK);
    }
    EXPECT_TRUE(sawDifferentMoveTimes);
}

TEST(ClockSamplingTest, SameSeedSameClock) {
    std::mt19937_64 first(42);
    std::mt19937_64 second(42);
    const ClockSampling sampling;
    for (int draw = 0; draw < 20; ++draw) {
        EXPECT_EQ(sample_clock(sampling, first), sample_clock(sampling, second));
    }
}

TEST(ClockSamplingTest, RejectsBoundsOutOfOrder) {
    ClockSampling sampling;
    EXPECT_NO_THROW(sampling.validate());
    sampling.startLeadMinMs = 0;
    EXPECT_THROW(sampling.validate(), std::invalid_argument);
    sampling = {};
    sampling.moveTimeMaxMs = sampling.moveTimeMinMs - 1;
    EXPECT_THROW(sampling.validate(), std::invalid_argument);
    sampling = {};
    sampling.levelStartFraction = 1.5;
    EXPECT_THROW(sampling.validate(), std::invalid_argument);
}
