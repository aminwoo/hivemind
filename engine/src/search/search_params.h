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
/// Q_init is clamped to Q_parent +/- 1 so constant value offsets do not saturate it.
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

/**
 * Discount factor per 100 plies remaining, favouring faster wins and delayed
 * losses: neuralValue is scaled by (1 - MOVES_LEFT_DISCOUNT * plies/100).
 *
 * The magnitude has to be large enough to break a near-tie between two moves
 * the value head rates alike, which is the only situation where a moves-left
 * signal is worth anything. At 0.005 the largest possible swing was 0.5% of the
 * value, so the head could not move a decision at all - it was wired up but
 * disconnected. At 0.05 a mate in ten plies beats one in sixty by ~0.02 of
 * value, comparable to the Q gaps MCTS actually decides on, without letting
 * distance-to-end override the evaluation itself. Bughouse is a mating race, so
 * this is the variant where finishing sooner is worth the most.
 *
 * Strength-affecting and unvalidated: sweep MovesLeftDiscountPermille in a
 * paired tournament before trusting the default.
 */
constexpr float MOVES_LEFT_DISCOUNT = 0.05f;

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

/**
 * How many joint plies below the retained subtree root are indexed for reuse.
 *
 * Bughouse needs the width. Our partner and both opponents keep moving while we
 * think, and a partner-board exchange - opponent moves, partner answers,
 * opponent moves again - is three joint plies, not the single predicted reply
 * that ponder retains.
 *
 * Every joint ply flips the team to move, and a search is only ever rooted at a
 * position where our own team is on move, so only the odd levels (plus the
 * subtree root itself, which is where the permanent brain starts) can ever be
 * adopted. Three therefore buys two usable levels; two would buy one.
 */
constexpr int TREE_REUSE_MAX_JOINT_PLIES = 3;

/// Hard cap on retained reuse candidates. Each one costs two FEN strings to
/// record, so the walk stops widening rather than delaying the next search.
constexpr size_t TREE_REUSE_MAX_CANDIDATES = 4096;

/// Bound the synchronous graph walk performed before workers restart. Nodes
/// beyond this cap remain reusable through ordinary edges; they simply are not
/// pre-seeded into the transposition table for this move.
constexpr size_t TREE_REUSE_REINDEX_MAX_NODES = 100000;

/**
 * Permanent brain: keep searching between moves instead of sitting idle.
 *
 * Ponder only fires when the GUI sends "go ponder" for the exact predicted
 * joint action, which with four asynchronous players almost never happens.
 * The permanent brain instead searches the position our own move creates, on
 * the engine's own initiative, spreading the work over every opponent reply
 * rather than betting on one. It stops at the next position, go, or stop.
 */
constexpr bool ENABLE_PERMANENT_BRAIN = true;

/// Safety rails on the background search, which has no clock of its own.
constexpr int PERMANENT_BRAIN_MAX_NODES = 500000;
constexpr double PERMANENT_BRAIN_MAX_MS = 60000.0;

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

/**
 * Maximum attacker moves for the root single-board forced-mate search.
 * The search is iteratively deepened from 2, so a shorter mate is preferred.
 */
constexpr int MATE_SEARCH_MAX_ATTACKER_MOVES = 5;

/**
 * Node budget (attacker moves plus defender replies tried) for the whole root
 * forced-mate pre-pass, shared across its deepening iterations.
 *
 * The pre-pass runs synchronously before MCTS, so it needs a cap of its own:
 * without one, an exposed king plus a full hand reaches multiple seconds. This
 * is the hard ceiling (~100 ms); Agent::search() scales it down further with the
 * budget of the search it precedes. Exhausting it only means "not proven" - the
 * position then goes to MCTS as usual, where the in-tree solver can still prove
 * the mate.
 */
// Safety net, not the operating limit: MATE_SEARCH_MAX_TIME_PERCENT is what
// bounds the pre-pass, and this used to bind first and stop a scan with time
// still on its clock. A root forced-loss proof needing ~4M probes - a
// cross-board feed mate - fits inside a 10s move and was cut off at 100k.
constexpr uint64_t MATE_SEARCH_NODE_BUDGET = 10000000;

/// Floor for the scaled budget, so even a very short search still gets a scan.
constexpr uint64_t MATE_SEARCH_MIN_NODE_BUDGET = 2000;

// A universal loss proof must cover every legal defense. Timed searches
// reserve enough cheap probes for this optional reverse proof. This applies to
// both clock states: find_root_forced_loss() models the appropriate sit rules,
// and skipping down-time roots suppresses exact losses that it can prove.
constexpr uint64_t FORCED_LOSS_MIN_TIMED_NODE_BUDGET = 40000;

// Capture-feed proofs are searched independently in both board directions and
// draw an allowance of their own, sized by the caller's budget. A fixed floor
// here would override the scaled budget above and tax short searches with the
// worst case of a long one, so the two directions simply split this share of
// the caller's allowance evenly - neither can starve the other.
constexpr uint64_t MATE_CAPTURE_FEED_NODE_BUDGET_PERCENT = 100;

// Wall clock the Fairy-Stockfish mate probe may use when the search has no move
// time to carve a pre-pass window out of, as in analysis. Timed searches hand
// it whatever is left of MATE_SEARCH_MAX_TIME_PERCENT instead.
constexpr int MATE_PROBE_UNTIMED_BUDGET_MS = 500;

// Longest mate the probe will stop for, and accept. It doubles as an evidence
// bound: a mate this short out of a pruned search is worth collapsing the root
// onto, where the mate in 32 at depth 63 that an unbounded probe once returned
// inside 400ms is not.
constexpr int MATE_PROBE_MAX_MATE_MOVES = 16;

// Answer the capture-feed scan's single-board mate question with
// Fairy-Stockfish instead of the checks-only scan.
//
// Off, measured. The feed scan keeps a candidate capture only when every
// opponent reply comes back "mated", so it needs verdicts that are exhaustive
// rather than merely deeper. The checks-only scan is restricted but complete
// within its restriction; a bounded Fairy-Stockfish search reports "no mate
// found" where it means "not yet", and one such reply discards the candidate.
// It is faster where it does apply - the suite's first position proves in 937ms
// against 1730ms - but costs 3 to 4 of 36 suite matches, equally at 2000, 6000
// and 20000 nodes per question, so it is not a budget problem.
//
// Falling back to the checks-only scan on a probe miss answers that objection
// and still costs matches, now on the clock rather than on verdicts: the
// probes spend the pre-pass window the exact scan needs, and the suite's
// four-move capture-feed mate goes missing at 1000ms - 2 of 42, both
// orientations of one position. Asking the probe for direct mates ahead of the
// exact scan is worse again, 8 of 42, because a bounded search answers with
// the mate it reached: a mate in 15 where the exact scan proves one in 2.
// The shape that would fit is still the hybrid: checks-only for the per-reply
// verdicts, and this only for the candidate that survives them.
constexpr bool ENABLE_STOCKFISH_MATE_SEARCH = false;

// Per-question ceiling for that probe, in the scan's own node currency, plus a
// wall-clock backstop.
constexpr uint64_t MATE_PROBE_FEED_NODE_BUDGET = 20000;
constexpr int MATE_PROBE_FEED_MAX_MS = 40;

// The root probe asks one question per board per search, so it may spend more.
constexpr uint64_t MATE_PROBE_ROOT_NODE_BUDGET = 8000000;

// Use the same evidence bound for accepting a probe mate and ending a move.
// A separate five-move early-exit cap made accepted mates in 6..16 wait for
// the deadline even though the probe had already stopped searching. These
// remain single-board search claims, not exact two-board proofs; the caller
// must still prefer an exact scan or a root already solved by MCTS.
constexpr bool mate_probe_can_end_search(int plyToMate) {
    return ENABLE_MATE_EARLY_EXIT && plyToMate > 0
        && plyToMate <= 2 * MATE_PROBE_MAX_MATE_MOVES - 1;
}


// The probe answers with a searched mate score rather than an exact proof, and
// it is the one part of the pre-pass that can be switched off without giving up
// the check-only scans, so it gets its own flag.
constexpr bool ENABLE_MATE_PROBE = true;

/**
 * Share of the move time the root proof pre-pass may occupy. Both clock states
 * run the winning scan and the bounded loss scan.
 *
 * Node budgets cannot enforce this on their own: a joint proof node costs
 * ~70us against ~1us for a single-board check node, a spread no single node
 * count covers. Sized so the slowest proof the scan still completes - a
 * seven-ply capture feed, ~210ms of a 1500ms move - fits with margin; a proof
 * that lands returns immediately and hands the rest of the move time back, so
 * the average cost is far below the cap.
 */
constexpr int MATE_SEARCH_MAX_TIME_PERCENT = 20;

// The winning scan and reverse (move-safety) scan share the work already done
// by concurrent MCTS, but the winning scan can consume its entire deadline.
// Give the reverse scan a small bounded tail in which to certify the current
// MCTS favourites. This is capped so long time controls do not turn it into a
// second multi-second search.
constexpr int ROOT_LOSS_EXTRA_TIME_PERCENT = 80;
constexpr int ROOT_LOSS_EXTRA_MAX_MS = 800;

/**
 * Tiny synchronous mate-in-one probe performed before neural workers start.
 *
 * Once a TensorRT batch has been dispatched it cannot be cancelled. Starting
 * MCTS first therefore makes even an immediately proven mate wait for that
 * batch to finish. Keep this probe deliberately small; if it does not finish,
 * the regular bounded root scan continues concurrently with MCTS.
 */
constexpr uint64_t IMMEDIATE_MATE_PREFLIGHT_NODE_BUDGET = 2000;
constexpr int IMMEDIATE_MATE_PREFLIGHT_MAX_MS = 5;

/// Probes between deadline samples. Keeps the clock read well under 1% of the
/// cheapest probe while still stopping a joint scan within a few milliseconds.
constexpr uint32_t MATE_SEARCH_TIME_CHECK_INTERVAL = 64;

/**
 * Conversion factors from the search's own budget to mate-search nodes.
 *
 * A mate-search node is a make/unmake plus a mate test, roughly fifty times
 * cheaper than an MCTS node with its network evaluation, so these keep the
 * pre-pass at a few percent of the search it precedes instead of a fixed cost
 * that can dominate a short one (self-play runs searches of a few hundred nodes).
 */
constexpr uint64_t MATE_SEARCH_NODES_PER_SEARCH_NODE = 10;
// Give a 1-second UCI search the full root-mate budget.  This prevents
// short searches from falling through to MCTS before the single-board solver
// has had enough work to prove tactical mates such as Qh5.
// Roughly what the scan gets through in a millisecond. At 50 it under-read
// throughput by ~40x, so the time-scaled budget was far smaller than the
// window it was scaling.
constexpr uint64_t MATE_SEARCH_NODES_PER_MILLISECOND = 2000;

/**
 * A joint proof node can enumerate and apply many board-move combinations,
 * making it substantially more expensive than one node in the optimized
 * single-board mate scan. Scale the shared pre-pass allowance accordingly.
 */
constexpr uint64_t MATE_JOINT_SEARCH_BUDGET_DIVISOR = 10;

// =============================================================================
// Progressive Widening Parameters
// =============================================================================

/// Allowed children: ceil(PW_COEFFICIENT * visits^PW_EXPONENT).
constexpr float PW_COEFFICIENT = 4.0f;
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

/// Larger learned joint pool used once per root, including promoted reused roots.
constexpr int ROOT_JOINT_POLICY_TOP_K = 32;

/// Scale applied to the learned residual compatibility score.
constexpr float JOINT_POLICY_RESIDUAL_SCALE = 1.0f;

struct RuntimeConfig {
    float cpuctInit = CPUCT_INIT;
    float cpuctBase = CPUCT_BASE;
    bool enableMCGS = ENABLE_MCGS;
    bool enableTranspositions = ENABLE_TRANSPOSITIONS;
    bool enableRootMateSearch = ENABLE_MATE_EARLY_EXIT;
    bool enableMateProbe = ENABLE_MATE_PROBE;
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
    int rootJointPolicyTopK = ROOT_JOINT_POLICY_TOP_K;
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
