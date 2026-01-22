#pragma once

#include "board.h"
#include "utils.h"
#include "Fairy-Stockfish/src/types.h"

/**
 * @brief Converts a bughouse board state into neural network input planes.
 *
 * This function maps the current board configuration into a series of float planes,
 * which can then be used as input for machine learning models or further processing.
 *
 * @param board The current bughouse board state.
 * @param inputPlanes Pointer to an array of floats where the resulting planes are stored.
 * @param us The color representing the current team's perspective.
 * @param pass A flag indicating whether to process additional input (e.g., passed moves).
 */
void board_to_planes(Board& board, float* inputPlanes, Stockfish::Color us, bool hasTimeAdvantage);
