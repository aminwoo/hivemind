#pragma once

#include <cmath>
#include <cstdint>

/**
 * @file search_params.h
 * @brief Centralized search hyperparameters for MCGS tuning.
 * 
 * All tunable MCGS (Monte Carlo Graph Search) parameters are defined here 
 * for easy experimentation. MCGS extends MCTS by using a transposition table
 * to detect when different move sequences reach the same position.
 * 
 * Default values are aligned with CrazyAra's search settings for consistency.
 */

namespace SearchParams {

// =============================================================================
// Batch MCGS Parameters (aligned with CrazyAra defaults)
// =============================================================================

/// Number of leaves to collect before batched neural network inference
/// CrazyAra default: 8
constexpr int BATCH_SIZE = 16;

/// Number of search threads to run in parallel
/// CrazyAra default: 2
constexpr int NUM_SEARCH_THREADS = 2;

// =============================================================================
// Virtual Loss Settings (aligned with CrazyAra)
// =============================================================================

/**
 * Virtual loss style for avoiding collisions during batched search:
 * - VIRTUAL_LOSS: Decreases Q-value as if a loss occurred (Q = (Q*n - 1) / (n + 1))
 * - VIRTUAL_VISIT: Only increments visit count without modifying Q-value
 * - VIRTUAL_OFFSET: Subtracts a small offset from Q-value
 * - VIRTUAL_MIX: Starts with VIRTUAL_VISIT, switches to VIRTUAL_LOSS after threshold
 * 
 * CrazyAra default: VIRTUAL_VISIT
 */
enum class VirtualStyle {
    VIRTUAL_LOSS,
    VIRTUAL_VISIT,
    VIRTUAL_OFFSET,
    VIRTUAL_MIX
};

/// Default virtual style (CrazyAra uses VIRTUAL_VISIT by default)
constexpr VirtualStyle VIRTUAL_STYLE = VirtualStyle::VIRTUAL_VISIT;

/// Threshold for switching from VIRTUAL_VISIT to VIRTUAL_LOSS in VIRTUAL_MIX mode
/// CrazyAra default: 1000
constexpr uint32_t VIRTUAL_MIX_THRESHOLD = 1000;

/// Strength of virtual offset when using VIRTUAL_OFFSET style
/// CrazyAra default: 0.001
constexpr double VIRTUAL_OFFSET_STRENGTH = 0.001;

// =============================================================================
// MCGS (Monte Carlo Graph Search) Parameters (aligned with CrazyAra)
// =============================================================================

/// Enable MCGS (Monte Carlo Graph Search) with transposition table
/// When true, positions reached through different paths share the same node
/// When false, MCGS is disabled and search behaves as traditional MCTS tree search
/// CrazyAra default: true
constexpr bool ENABLE_MCGS = true;

/// Enable transposition table for graph-based search (only used if ENABLE_MCGS is true)
/// When true, positions reached through different paths share the same node
constexpr bool ENABLE_TRANSPOSITIONS = true;

/// Initial capacity for transposition table (number of positions)
/// Higher values reduce rehashing overhead but use more memory
constexpr size_t TT_INITIAL_CAPACITY = 100000;

/// Maximum transposition table size (0 = unlimited)
/// Prevents unbounded memory growth in long games
/// CrazyAra MAX_HASH_SIZE: 100000000
constexpr size_t TT_MAX_SIZE = 0;

/// Minimum visit count for a node to be used as a transposition
/// Higher values ensure more reliable Q-value estimates before sharing
constexpr int TT_MIN_VISITS = 0;

// =============================================================================
// PUCT (Polynomial Upper Confidence Trees) Parameters (aligned with CrazyAra)
// =============================================================================

/// Initial exploration constant for PUCT formula
/// Higher values encourage more exploration
/// CrazyAra default: 2.5
constexpr float CPUCT_INIT = 2.5f;

/// Base value for dynamic CPUCT scaling
/// CPUCT = log((N + CPUCT_BASE + 1) / CPUCT_BASE) + CPUCT_INIT
/// CrazyAra default: 19652
constexpr float CPUCT_BASE = 19652.0f;

// =============================================================================
// First Play Urgency (FPU) Parameters
// =============================================================================

/**
 * FPU Strategy: CrazyAra uses Q_INIT = -1.0 for unvisited children.
 * 
 * Your engine uses parent-relative FPU: FPU = parent_Q - FPU_REDUCTION
 * This is a valid alternative approach (used by Lc0 and others).
 * 
 * For consistency with CrazyAra-style behavior:
 * - Set FPU_REDUCTION to a high value (like 1.0) to start unvisited nodes pessimistically
 * - Or keep current value for Lc0-style parent-relative FPU
 * 
 * Current setting: 0.4 (Lc0-style parent-relative FPU)
 * CrazyAra equivalent: Would be approximately Q_INIT = -1.0 (fixed pessimistic)
 */
constexpr float FPU_REDUCTION = 0.4f;

// =============================================================================
// Draw Contempt Parameters
// =============================================================================

/**
 * Draw contempt: Penalty applied to draw evaluations to encourage playing for wins.
 * 
 * Positive values make the engine avoid draws (treat them slightly as losses).
 * A value of 0.05 means draws are evaluated as -0.05 instead of 0.0.
 * 
 * This encourages more aggressive play and prevents the engine from being
 * overly content with drawn positions.
 * 
 * Typical range: 0.0 to 0.15
 * - 0.0: No contempt (draws are neutral)
 * - 0.05: Light contempt (slightly prefer wins over draws)
 * - 0.10: Moderate contempt (more aggressive)
 */
constexpr float DRAW_CONTEMPT = 0.15f;

// =============================================================================
// Progressive Widening Parameters
// =============================================================================

/// Coefficient for progressive widening formula
/// m = ceil(PW_COEFFICIENT * n^PW_EXPONENT)
/// where m = allowed children, n = visit count
constexpr float PW_COEFFICIENT = 1.0f;

/// Exponent for progressive widening formula
/// Lower values slow down the expansion rate
constexpr float PW_EXPONENT = 0.3f;

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Calculates dynamic CPUCT based on parent visit count.
 * 
 * Uses logarithmic scaling similar to AlphaZero/Lc0/CrazyAra.
 * Formula: CPUCT = log((N + CPUCT_BASE + 1) / CPUCT_BASE) + CPUCT_INIT
 * 
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

/**
 * @brief Gets the effective virtual style based on visit count.
 * 
 * CrazyAra uses VIRTUAL_MIX to switch from VIRTUAL_VISIT to VIRTUAL_LOSS
 * after a certain threshold. This helper implements that logic.
 * 
 * @param visits Current visit count of the child
 * @return The virtual style to use
 */
inline VirtualStyle get_virtual_style(uint32_t visits) {
    if (VIRTUAL_STYLE == VirtualStyle::VIRTUAL_MIX) {
        if (visits > VIRTUAL_MIX_THRESHOLD) {
            return VirtualStyle::VIRTUAL_LOSS;
        }
        return VirtualStyle::VIRTUAL_VISIT;
    }
    return VIRTUAL_STYLE;
}

/**
 * @brief Apply virtual loss/visit to Q-value during selection.
 * 
 * CrazyAra virtual loss formula:
 * - VIRTUAL_LOSS: Q = (Q * n - 1) / (n + 1)  (treats as if a loss occurred)
 * - VIRTUAL_VISIT: Only increment visit count, Q unchanged
 * - VIRTUAL_OFFSET: Q -= VIRTUAL_OFFSET_STRENGTH
 * 
 * @param currentQ Current Q-value
 * @param visits Current visit count
 * @param style Virtual style to apply
 * @return Updated Q-value after virtual loss
 */
inline float apply_virtual_loss(float currentQ, uint32_t visits, VirtualStyle style) {
    switch (style) {
    case VirtualStyle::VIRTUAL_LOSS:
        return static_cast<float>((static_cast<double>(currentQ) * visits - 1.0) / (visits + 1.0));
    case VirtualStyle::VIRTUAL_OFFSET:
        return currentQ - static_cast<float>(VIRTUAL_OFFSET_STRENGTH);
    case VirtualStyle::VIRTUAL_VISIT:
    case VirtualStyle::VIRTUAL_MIX:
    default:
        return currentQ;  // Q unchanged, only visit count changes
    }
}

/**
 * @brief Revert virtual loss during backup.
 * 
 * @param currentQ Current Q-value (with virtual loss applied)
 * @param visits Current visit count (including virtual visit)
 * @param style Virtual style that was applied
 * @return Q-value with virtual loss reverted
 */
inline float revert_virtual_loss(float currentQ, uint32_t visits, VirtualStyle style) {
    switch (style) {
    case VirtualStyle::VIRTUAL_LOSS:
        if (visits <= 1) return currentQ;
        return static_cast<float>((static_cast<double>(currentQ) * visits + 1.0) / (visits - 1.0));
    case VirtualStyle::VIRTUAL_OFFSET:
        return currentQ + static_cast<float>(VIRTUAL_OFFSET_STRENGTH);
    case VirtualStyle::VIRTUAL_VISIT:
    case VirtualStyle::VIRTUAL_MIX:
    default:
        return currentQ;  // Q was unchanged
    }
}

} // namespace SearchParams
