#pragma once

#include <cmath>

/**
 * @file search_params.h
 * @brief Centralized search hyperparameters for MCGS tuning.
 * 
 * All tunable MCGS (Monte Carlo Graph Search) parameters are defined here 
 * for easy experimentation. MCGS extends MCTS by using a transposition table
 * to detect when different move sequences reach the same position.
 */

namespace SearchParams {

// =============================================================================
// Batch MCGS Parameters
// =============================================================================

/// Number of leaves to collect before batched neural network inference
constexpr int BATCH_SIZE = 16;

/// Number of search threads to run in parallel
constexpr int NUM_SEARCH_THREADS = 2;

/// Virtual loss amount applied during batch selection to reduce collisions
constexpr int VIRTUAL_LOSS = 1;

// =============================================================================
// MCGS (Monte Carlo Graph Search) Parameters
// =============================================================================

/// Enable transposition table for graph-based search
/// When true, positions reached through different paths share the same node
constexpr bool ENABLE_TRANSPOSITIONS = true;

/// Initial capacity for transposition table (number of positions)
/// Higher values reduce rehashing overhead but use more memory
constexpr size_t TT_INITIAL_CAPACITY = 100000;

/// Maximum transposition table size (0 = unlimited)
/// Prevents unbounded memory growth in long games
constexpr size_t TT_MAX_SIZE = 0;

/// Minimum visit count for a node to be used as a transposition
/// Higher values ensure more reliable Q-value estimates before sharing
constexpr int TT_MIN_VISITS = 0;

// =============================================================================
// PUCT (Polynomial Upper Confidence Trees) Parameters
// =============================================================================

/// Initial exploration constant for PUCT formula
/// Higher values encourage more exploration
constexpr float CPUCT_INIT = 2.5f;

/// Base value for dynamic CPUCT scaling
/// CPUCT = log((N + CPUCT_BASE + 1) / CPUCT_BASE) + CPUCT_INIT
constexpr float CPUCT_BASE = 19652.0f;

// =============================================================================
// First Play Urgency (FPU) Parameters
// =============================================================================

/// Reduction from parent Q-value for unvisited children
/// FPU = parent_Q - FPU_REDUCTION, clamped to [-1, 1]
/// Lower values make unvisited nodes less attractive
constexpr float FPU_REDUCTION = 0.4f;

// =============================================================================
// Progressive Widening Parameters
// =============================================================================

/// Coefficient for progressive widening formula
/// m = ceil(PW_COEFFICIENT * n^PW_EXPONENT)
/// where m = allowed children, n = visit count
constexpr float PW_COEFFICIENT = 2.0f;

/// Exponent for progressive widening formula
/// Lower values slow down the expansion rate
constexpr float PW_EXPONENT = 0.5f;

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Calculates dynamic CPUCT based on parent visit count.
 * 
 * Uses logarithmic scaling similar to AlphaZero/Lc0.
 * @param totalVisits Parent node's total visit count
 * @return Dynamic CPUCT value
 */
inline float get_cpuct(float totalVisits) {
    return std::log((totalVisits + CPUCT_BASE + 1.0f) / CPUCT_BASE) + CPUCT_INIT;
}

/**
 * @brief Calculates the number of allowed children based on progressive widening.
 * 
 * Formula: m = ceil(PW_COEFFICIENT * n^PW_EXPONENT)
 * @param visitCount Current visit count of the node
 * @return Number of children allowed to be expanded
 */
inline int get_allowed_children(int visitCount) {
    if (visitCount <= 0) return 1;
    return static_cast<int>(std::ceil(PW_COEFFICIENT * std::pow(static_cast<float>(visitCount), PW_EXPONENT)));
}

} // namespace SearchParams
