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
 * Defaults use CrazyAra where the search semantics transfer directly, with
 * Bughouse-specific settings for joint actions, passing, and terminal handling.
 */

namespace SearchParams {

// =============================================================================
// Batch MCGS Parameters
// =============================================================================

/// Number of leaves to collect before batched neural network inference
/// CrazyAra default: 8
constexpr int BATCH_SIZE = 8;

/// Number of search threads to run in parallel per inference engine
/// Optimized for multi-core scaling with batched TensorRT engine
constexpr int NUM_SEARCH_THREADS = 4;

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

/// Full loss strongly separates two batched workers that can otherwise reach
/// the same canonical MCGS node through different incoming joint-action edges.
/// CrazyAra defaults to VIRTUAL_VISIT for its less transposition-heavy search.
constexpr VirtualStyle VIRTUAL_STYLE = VirtualStyle::VIRTUAL_LOSS;

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

/// Maximum transposition table size
/// Prevents unbounded memory growth in long games
constexpr size_t TT_MAX_SIZE = 1000000;

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

/// Enable dynamic FPU based on parent node's Q-value.
/// CrazyAra/Lc0 default: true
constexpr bool ENABLE_DYNAMIC_FPU = true;

/// Reduction from parent Q-value for unvisited nodes scaled by sqrt of visited policy.
/// Q_init = clamp(Q_parent - FPU_REDUCTION * sqrt(sum_visited_priors), -1.0, 1.0)
/// Lc0/CrazyAra default: 1.0
constexpr float FPU_REDUCTION = 1.0f;

/// Initial Q-value for an unvisited edge when network values use [-1, +1] (or fallback).
constexpr float Q_INIT = -1.0f;

// =============================================================================
// Auxiliary Neural Network Head Parameters (WDL & Moves-Left)
// =============================================================================

/// Enable WDL-based expected value evaluation with dynamic draw contempt.
constexpr bool ENABLE_WDL_EVAL = true;

/// Weight of WDL expected value vs direct scalar value (1.0 = pure WDL, 0.0 = pure scalar)
constexpr float WDL_VALUE_WEIGHT = 0.25f;

/// Discount factor per 100 plies remaining to favor faster wins and delayed losses
constexpr float MOVES_LEFT_DISCOUNT = 0.005f;

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
constexpr float DRAW_CONTEMPT = 0.0f;

// =============================================================================
// Q-Value Weighted Move Selection Parameters (CrazyAra 2019)
// =============================================================================

/**
 * Q-value veto: If best Q-value move differs from most-visited move by more than
 * this delta, swap their visit counts to promote the higher-Q move.
 * 
 * This prevents cases where a clearly better move hasn't received enough visits
 * to become the most-visited child.
 * 
 * CrazyAra default: 0.4
 * Set to 0.0 to disable Q-value veto.
 */
constexpr float Q_VETO_DELTA = 0.4f;

/**
 * Q-value weight: Transfers probability mass from most-visited to second-best
 * move proportional to Q-value difference.
 * 
 * When the second-best Q-value is higher than the best-visited move:
 * policy[secondBest] += qDiff * Q_VALUE_WEIGHT * policy[bestVisited]
 * 
 * CrazyAra default: 1.0
 * Set to 0.0 to disable Q-value weighting.
 */
constexpr float Q_VALUE_WEIGHT = 1.0f;

// =============================================================================
// Tree Reuse Parameters
// =============================================================================

/**
 * Enable tree reuse: Preserve search tree between moves.
 * 
 * When enabled, the engine stores pointers to likely next roots:
 * - ownNextRoot: The most-visited child (our expected move)
 * - opponentsNextRoot: Opponent's most-visited response
 * 
 * On the next search, if the position matches, the subtree is reused.
 * A candidate is only adopted when its recorded board signature matches the
 * live board exactly, so retained edges (notably drops) stay legal even after a
 * UCI position reconstruction discards repetition history.
 */
constexpr bool ENABLE_TREE_REUSE = true;

// =============================================================================
// Early Stopping and Time Management Parameters
// =============================================================================

/**
 * Enable early stopping: Stop search when best move has insurmountable lead.
 * 
 * When the second-best move cannot catch up to the best move even with
 * all remaining search time, stop early to save time.
 * 
 * Condition: (secondMax + remainingTime * NPS) * EARLY_STOP_FACTOR < firstMax
 * 
 * CrazyAra default: false
 */
constexpr bool ENABLE_EARLY_STOPPING = true;

/// Factor for early stopping comparison (best move must retain this multiple of projected visits)
/// CrazyAra uses 2.0
constexpr float EARLY_STOP_FACTOR = 2.0f;

/**
 * Enable early exit when position is solved with proven mate.
 * 
 * Stop search immediately when:
 * - Root node is proven WIN (we have forced mate)
 * - Best child is proven LOSS for opponent (we have forced mate via that move)
 * 
 * This saves time in proven terminal/mate positions.
 */
constexpr bool ENABLE_MATE_EARLY_EXIT = true;

/**
 * Enable dynamic time extension: Extend search when evaluation is falling
 * or when the leading move changes late in search.
 * 
 * CrazyAra NPS time-manager default: false
 */
constexpr bool ENABLE_TIME_EXTENSION = true;

/// Maximum number of time extensions per move (to prevent infinite extension)
constexpr int MAX_TIME_EXTENSIONS = 2;

/// Minimum evaluation drop in normalized Q units to trigger time extension
/// Smaller drops are normal fluctuation, not worth extending for
constexpr float TIME_EXTENSION_THRESHOLD = 0.05f;

/// Factor to multiply remaining time when extending (1.5 = 50% more time)
constexpr float TIME_EXTENSION_FACTOR = 1.5f;

/// Fraction of initial move time elapsed before best-move changes trigger instability extension
constexpr float INSTABILITY_TIME_FRACTION = 0.4f;

// =============================================================================
// MCTS Solver Parameters
// =============================================================================

/**
 * Enable MCTS solver: Propagate proven WIN/LOSS/DRAW states up the tree.
 * 
 * When a terminal node is reached:
 * - WIN for opponent (they got mated) → mark parent as WIN
 * - LOSS for us (we got mated) → mark parent as candidate LOSS
 * - Parent is WIN if any child is LOSS (opponent)
 * - Parent is LOSS if all children are WIN (opponent)
 * 
 * CrazyAra default: false. Bughouse enables it because forced mating sequences
 * are common and the solver only proves losses after all joint actions exist.
 */
constexpr bool ENABLE_MCTS_SOLVER = true;

// =============================================================================
// Progressive Widening Parameters
// =============================================================================

/// Allowed children: ceil(PW_COEFFICIENT * visits^PW_EXPONENT).
constexpr float PW_COEFFICIENT = 2.0f;
constexpr float ROOT_PW_COEFFICIENT = 4.0f;
constexpr float PW_EXPONENT = 0.4f;

// =============================================================================
// Solver-aware Gumbel root search
// =============================================================================

/// Use Gumbel sequential halving at the root while retaining ordinary PUCT below it.
constexpr bool ENABLE_GUMBEL_ROOT_SEARCH = false;

/// Number of high-policy joint actions assigned independent Gumbel samples.
/// Tactical benchmarks favor a tiny tournament before returning to PUCT.
constexpr int ROOT_GUMBEL_POOL_SIZE = 2;

/// Number of actions in the first sequential-halving tournament.
constexpr int ROOT_GUMBEL_INITIAL_CANDIDATES = 1;

/// New actions introduced after an inconclusive tournament finalist.
constexpr int ROOT_GUMBEL_REPLENISHMENT = 1;

/// Relative influence of backed-up root Q in the Gumbel ranking.
constexpr float ROOT_GUMBEL_VALUE_SCALE = 2.0f;

/// Maximum equal-allocation quota for a single sequential-halving round.
constexpr int ROOT_GUMBEL_MAX_ROUND_VISITS = 2;

/// Number of marginal candidates per board rescored by an optional joint head.
constexpr int JOINT_POLICY_TOP_K = 8;

/// Scale applied to the learned residual compatibility score.
constexpr float JOINT_POLICY_RESIDUAL_SCALE = 1.0f;

struct RuntimeConfig {
    float cpuctInit = CPUCT_INIT;
    float cpuctBase = CPUCT_BASE;
    bool enableMCGS = ENABLE_MCGS;
    bool enableTranspositions = ENABLE_TRANSPOSITIONS;
    float drawContempt = DRAW_CONTEMPT;
    bool enableDynamicFpu = ENABLE_DYNAMIC_FPU;
    float fpuReduction = FPU_REDUCTION;
    bool enableWdlEval = ENABLE_WDL_EVAL;
    float wdlValueWeight = WDL_VALUE_WEIGHT;
    float movesLeftDiscount = MOVES_LEFT_DISCOUNT;
    bool enableTimeExtension = ENABLE_TIME_EXTENSION;
    float pwCoefficient = PW_COEFFICIENT;
    float rootPwCoefficient = ROOT_PW_COEFFICIENT;
    float pwExponent = PW_EXPONENT;
    float qValueWeight = Q_VALUE_WEIGHT;
    float qVetoDelta = Q_VETO_DELTA;
    float rootDirichletAlpha = 0.0f;
    float rootDirichletEpsilon = 0.0f;
    uint64_t rootNoiseSeed = 0;
    bool enableGumbelRootSearch = ENABLE_GUMBEL_ROOT_SEARCH;
    int rootGumbelPoolSize = ROOT_GUMBEL_POOL_SIZE;
    int rootGumbelInitialCandidates = ROOT_GUMBEL_INITIAL_CANDIDATES;
    int rootGumbelReplenishment = ROOT_GUMBEL_REPLENISHMENT;
    float rootGumbelValueScale = ROOT_GUMBEL_VALUE_SCALE;
    int rootGumbelMaxRoundVisits = ROOT_GUMBEL_MAX_ROUND_VISITS;
    int jointPolicyTopK = JOINT_POLICY_TOP_K;
    float jointPolicyResidualScale = JOINT_POLICY_RESIDUAL_SCALE;
};

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
inline float get_cpuct(float totalVisits, float init = CPUCT_INIT, float base = CPUCT_BASE) {
    return std::log((totalVisits + base + 1.0f) / base) + init;
}

inline int get_allowed_children(int visitCount,
                                float coefficient = PW_COEFFICIENT,
                                float exponent = PW_EXPONENT) {
    if (visitCount <= 0) return 1;
    return static_cast<int>(std::ceil(
        coefficient * std::pow(static_cast<float>(visitCount), exponent)));
}

inline bool has_insurmountable_visit_lead(float bestVisits,
                                          float projectedSecondVisits,
                                          float factor = EARLY_STOP_FACTOR) {
    return projectedSecondVisits * factor < bestVisits;
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
} // namespace SearchParams
