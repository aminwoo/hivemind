#pragma once

#include "nn/backend_compat.h"

#include "environment/board.h"
#include "environment/time_state.h"
#include "common/utils.h"
#include "Fairy-Stockfish/src/types.h"

/**
 * @brief The lead plane: @p team's lead over its diagonal opponents, counted
 * in the moves the leading team can afford to wait.
 *
 * sign(lead) * min(1, log2(1 + moves) / 10), so the value saturates at about
 * a thousand moves. A legacy flag is a permanent lead or deficit, +1 or -1;
 * the old networks' training data maps the same way.
 */
float clock_lead_plane_value(const TimeState& time, Stockfish::Color team);

/**
 * @brief The speed plane: how much slower @p team acts than its opponents,
 * log2(own move time / their move time) / 5, clamped to [-1, 1]. Zero for a
 * legacy flag, which carries no speeds.
 */
float clock_speed_plane_value(const TimeState& time, Stockfish::Color team);

/**
 * @brief Converts a bughouse board state into neural network input planes.
 *
 * Each board's block holds 37 position planes (the sit right among them)
 * and then the lead and speed planes. Single-board variants leave the time
 * planes at zero.
 *
 * @param board The current bughouse board state.
 * @param inputPlanes Pointer to an array of floats where the resulting planes are stored.
 * @param teamSide The color representing the current team's perspective.
 * @param time The clock at this position, from which the team's sit right,
 *        lead and speed come.
 */
void board_to_planes(Board& board, float* inputPlanes, Stockfish::Color teamSide,
                     const TimeState& time);
/** Legacy form: @p hasTimeAdvantage is a permanent flag for @p teamSide. */
void board_to_planes(Board& board, float* inputPlanes, Stockfish::Color teamSide, bool hasTimeAdvantage);

#if defined(HIVEMIND_BACKEND_TENSORRT) || defined(HIVEMIND_ORT_FP16)
void board_to_planes(Board& board, __half* inputPlanes, Stockfish::Color teamSide,
                     const TimeState& time);
void board_to_planes(Board& board, __half* inputPlanes, Stockfish::Color teamSide, bool hasTimeAdvantage);
#endif
