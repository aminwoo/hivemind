#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

#include "environment/time_state.h"

/**
 * @brief How self-play and tournament games draw their simulated clocks.
 *
 * A game starts level with probability levelStartFraction. Otherwise the
 * white team's lead is drawn log-uniformly from [startLeadMinMs,
 * startLeadMaxMs] with a random sign. Each team's move time is drawn
 * log-uniformly from [moveTimeMinMs, moveTimeMaxMs]; set the bounds equal for
 * a fixed value.
 */
struct ClockSampling {
    double levelStartFraction = 0.5;
    int32_t startLeadMinMs = 100;
    int32_t startLeadMaxMs = 10'000;
    int32_t moveTimeMinMs = TimeState::DEFAULT_MOVE_TIME_MS;
    int32_t moveTimeMaxMs = 3'000;

    void validate() const {
        if (!(levelStartFraction >= 0.0 && levelStartFraction <= 1.0)) {
            throw std::invalid_argument("level start fraction must be in [0, 1]");
        }
        if (startLeadMinMs < 1 || startLeadMaxMs < startLeadMinMs
            || startLeadMaxMs > TimeState::MAX_LEAD_MS) {
            throw std::invalid_argument(
                "start lead bounds must satisfy 1 <= min <= max");
        }
        if (moveTimeMinMs < 1 || moveTimeMaxMs < moveTimeMinMs
            || moveTimeMaxMs > 3'600'000) {
            throw std::invalid_argument(
                "move time bounds must satisfy 1 <= min <= max <= 3600000");
        }
    }
};

/** A value drawn log-uniformly from [minimum, maximum]. */
inline int32_t sample_log_uniform(int32_t minimum, int32_t maximum,
                                  std::mt19937_64& random) {
    if (minimum >= maximum) {
        return minimum;
    }
    std::uniform_real_distribution<double> exponent(
        std::log(static_cast<double>(minimum)),
        std::log(static_cast<double>(maximum)));
    const double value = std::round(std::exp(exponent(random)));
    return static_cast<int32_t>(std::clamp<double>(value, minimum, maximum));
}

/** The white team's starting lead in milliseconds. */
inline int32_t sample_start_lead(const ClockSampling& sampling,
                                 std::mt19937_64& random) {
    if (std::bernoulli_distribution(sampling.levelStartFraction)(random)) {
        return 0;
    }
    const int32_t magnitude = sample_log_uniform(
        sampling.startLeadMinMs, sampling.startLeadMaxMs, random);
    return std::bernoulli_distribution(0.5)(random) ? magnitude : -magnitude;
}

/** A game's clock: a starting lead and a move time for each team. */
inline TimeState sample_clock(const ClockSampling& sampling,
                              std::mt19937_64& random) {
    const int32_t whiteLeadMs = sample_start_lead(sampling, random);
    const int32_t whiteMoveTimeMs = sample_log_uniform(
        sampling.moveTimeMinMs, sampling.moveTimeMaxMs, random);
    const int32_t blackMoveTimeMs = sample_log_uniform(
        sampling.moveTimeMinMs, sampling.moveTimeMaxMs, random);
    return TimeState::clocked(
        Stockfish::WHITE, whiteLeadMs, whiteMoveTimeMs, blackMoveTimeMs);
}
