#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <numeric>
#include "Fairy-Stockfish/src/types.h"
#include "search/search_params.h"

inline bool is_double_sit_legal(bool teamHasTimeAdvantage,
                                bool boardAOnTurn,
                                bool boardBOnTurn) {
    return teamHasTimeAdvantage && (boardAOnTurn != boardBOnTurn);
}

/**
 * @brief Legality of passing on one board while the partner board moves.
 *
 * Without a time advantage and with both boards on turn, sitting on one board
 * is only allowed when the partner board captures.
 */
inline bool is_single_pass_legal(bool teamHasTimeAdvantage,
                                 bool boardAOnTurn,
                                 bool boardBOnTurn,
                                 bool partnerMoveIsCapture) {
    return teamHasTimeAdvantage
        || !(boardAOnTurn && boardBOnTurn)
        || partnerMoveIsCapture;
}

/**
 * @brief Turn/time context needed to judge whether a joint action is legal.
 *
 * `boardXCanMove` is true only when that board is on turn and has at least one
 * legal move, so a pass there is a choice rather than a forced wait.
 */
struct JointActionRules {
    bool boardAOnTurn = false;
    bool boardBOnTurn = false;
    bool teamHasTimeAdvantage = false;
    bool boardACanMove = false;
    bool boardBCanMove = false;
};

// Hash function for pair<size_t, size_t> used in visited set
struct PairHash {
    size_t operator()(const std::pair<size_t, size_t>& p) const {
        // Combine hashes using bit mixing for better distribution
        return std::hash<size_t>()(p.first) ^ (std::hash<size_t>()(p.second) << 16);
    }
};

/**
 * @brief Represents a joint action candidate for Bughouse MCTS.
 * 
 * In Bughouse, the action space is the Cartesian product of moves on Board 0 and Board 1.
 * moveA is always on board 0, moveB is always on board 1.
 * This structure stores a pair of moves along with their joint prior probability.
 */
struct JointActionCandidate {
    Stockfish::Move moveA;         // Move on board 0
    Stockfish::Move moveB;         // Move on board 1
    float priorA;                  // Prior probability for move A
    float priorB;                  // Prior probability for move B
    float jointPrior;              // P(a|s) = P_A(a_A|s) * P_B(a_B|s)
    float expansionPriority;       // Candidate ordering (same as jointPrior)
    float gumbelScore;              // Root-only log-prior plus independent Gumbel noise
    size_t idxA;                   // Index in sorted actionsA
    size_t idxB;                   // Index in sorted actionsB

    JointActionCandidate() 
        : moveA(Stockfish::MOVE_NONE), moveB(Stockfish::MOVE_NONE),
          priorA(0.0f), priorB(0.0f), jointPrior(0.0f), expansionPriority(0.0f),
          gumbelScore(-std::numeric_limits<float>::infinity()),
          idxA(0), idxB(0) {}

    JointActionCandidate(Stockfish::Move mA, float pA, size_t iA,
                         Stockfish::Move mB, float pB, size_t iB,
                         const JointActionRules& rules = JointActionRules(),
                         bool moveAIsCapture = false,
                         bool moveBIsCapture = false)
        : moveA(mA), moveB(mB),
          priorA(pA), priorB(pB),
          idxA(iA), idxB(iB) {

        const bool sitsOnA = mA == Stockfish::MOVE_NONE;
        const bool sitsOnB = mB == Stockfish::MOVE_NONE;
        bool isInvalidSit = false;
        if (sitsOnA && sitsOnB) {
            isInvalidSit = !is_double_sit_legal(
                rules.teamHasTimeAdvantage, rules.boardAOnTurn, rules.boardBOnTurn);
        } else if (sitsOnA && rules.boardACanMove) {
            isInvalidSit = !is_single_pass_legal(
                rules.teamHasTimeAdvantage, rules.boardAOnTurn, rules.boardBOnTurn,
                moveBIsCapture);
        } else if (sitsOnB && rules.boardBCanMove) {
            isInvalidSit = !is_single_pass_legal(
                rules.teamHasTimeAdvantage, rules.boardAOnTurn, rules.boardBOnTurn,
                moveAIsCapture);
        }

        jointPrior = isInvalidSit ? -1.0f : pA * pB;
        expansionPriority = jointPrior;
        gumbelScore = std::log(std::max(jointPrior, 1.0e-30f));
    }

    // For max-heap comparison (forcing expansion priority first)
    bool operator<(const JointActionCandidate& other) const {
        return expansionPriority < other.expansionPriority;
    }
};

/**
 * @brief Lazy priority queue generator for joint action candidates.
 * 
 * Instead of generating all N*M pairs upfront, this class uses a max-heap
 * to lazily generate pairs in order of decreasing joint prior P_A * P_B.
 * 
 * Algorithm:
 * 1. Sort moves by prior (descending) on each board
 * 2. Start with (0,0) - the best pair from both boards
 * 3. When popping (i,j), push (i+1,j) and (i,j+1) if not already visited
 * 4. This lazily generates joint moves by descending joint prior
 */
class JointCandidateGenerator {
private:
    // Sorted actions and priors for each board
    std::vector<Stockfish::Move> sortedActionsA;
    std::vector<Stockfish::Move> sortedActionsB;
    std::vector<float> sortedPriorsA;
    std::vector<float> sortedPriorsB;
    std::vector<uint8_t> sortedCapturesA;
    std::vector<uint8_t> sortedCapturesB;
    std::vector<float> sortedJointFactorsA;
    std::vector<float> sortedJointFactorsB;
    // Max-heap for lazy generation
    std::priority_queue<JointActionCandidate> heap;
    
    // Track visited (i,j) pairs to avoid duplicates - O(1) lookup
    std::unordered_set<std::pair<size_t, size_t>, PairHash> visited;
    std::unordered_set<std::pair<size_t, size_t>, PairHash> jointPoolKeys;
    std::unordered_set<std::pair<size_t, size_t>, PairHash> generatedCandidateKeys;
    
    // Cache of already-generated candidates (for random access)
    std::vector<JointActionCandidate> generatedCandidates;
    std::vector<JointActionCandidate> jointPolicyCandidates;
    size_t nextJointPolicyCandidate = 0;

    // Root-only perturbed candidate pool. These are removed from the ordinary
    // factorized frontier, sorted by log-prior plus Gumbel noise, and emitted
    // before the lazy fallback resumes.
    std::vector<JointActionCandidate> gumbelCandidates;
    size_t nextGumbelCandidate = 0;
    size_t jointFactorRank = 0;
    
    // Turn, time and pass context used to reject illegal joint actions
    JointActionRules rules;

    void pushCandidate(size_t idxA, size_t idxB) {
        if (idxA >= sortedActionsA.size() || idxB >= sortedActionsB.size()) {
            return;
        }
        
        auto key = std::make_pair(idxA, idxB);
        if (visited.find(key) != visited.end()) {
            return;
        }
        visited.insert(key);

        if (jointPoolKeys.contains(key) || generatedCandidateKeys.contains(key)) {
            pushCandidate(idxA + 1, idxB);
            pushCandidate(idxA, idxB + 1);
            return;
        }
        
        // Create candidate to check if it's valid
        JointActionCandidate candidate(
            sortedActionsA[idxA], sortedPriorsA[idxA], idxA,
            sortedActionsB[idxB], sortedPriorsB[idxB], idxB,
            rules,
            sortedCapturesA[idxA] != 0,
            sortedCapturesB[idxB] != 0
        );
        
        // Only push valid candidates (jointPrior >= 0)
        if (candidate.jointPrior >= 0.0f) {
            heap.push(candidate);
        } else {
            // Invalid candidate (e.g., double pass without time advantage)
            // We still need to explore its successors to avoid missing valid moves
            // Recursively try the adjacent candidates
            pushCandidate(idxA + 1, idxB);
            pushCandidate(idxA, idxB + 1);
        }
    }

    JointActionCandidate popFactorizedCandidate() {
        JointActionCandidate best = heap.top();
        heap.pop();
        pushCandidate(best.idxA + 1, best.idxB);
        pushCandidate(best.idxA, best.idxB + 1);
        return best;
    }

    JointActionCandidate popBestUnperturbedCandidate() {
        const bool hasPolicyCandidate =
            nextJointPolicyCandidate < jointPolicyCandidates.size();
        if (hasPolicyCandidate &&
            (heap.empty() ||
             jointPolicyCandidates[nextJointPolicyCandidate].expansionPriority
                 >= heap.top().expansionPriority)) {
            return jointPolicyCandidates[nextJointPolicyCandidate++];
        }
        return popFactorizedCandidate();
    }

    void rebuildCandidateFrontier(size_t topK, float residualScale) {
        while (!heap.empty()) heap.pop();
        visited.clear();
        jointPoolKeys.clear();
        generatedCandidateKeys.clear();
        jointPolicyCandidates.clear();
        nextJointPolicyCandidate = 0;
        gumbelCandidates.clear();
        nextGumbelCandidate = 0;

        std::unordered_map<std::pair<size_t, size_t>, size_t, PairHash>
            generatedIndices;
        generatedIndices.reserve(generatedCandidates.size());
        for (size_t index = 0; index < generatedCandidates.size(); ++index) {
            JointActionCandidate& candidate = generatedCandidates[index];
            const auto key = std::make_pair(candidate.idxA, candidate.idxB);
            generatedCandidateKeys.emplace(key);
            generatedIndices.emplace(key, index);

            // Promotion may enlarge or shrink the learned pool. Restore every
            // expanded edge's factorized prior before applying the new pool so
            // old root scores cannot leak across configurations.
            candidate.jointPrior =
                sortedPriorsA[candidate.idxA] * sortedPriorsB[candidate.idxB];
            candidate.expansionPriority = candidate.jointPrior;
            candidate.gumbelScore = std::log(
                std::max(candidate.jointPrior, 1.0e-30f));
        }

        const bool hasJointFactors = jointFactorRank > 0 && topK > 0
            && sortedJointFactorsA.size()
                == sortedActionsA.size() * jointFactorRank
            && sortedJointFactorsB.size()
                == sortedActionsB.size() * jointFactorRank;
        if (!hasJointFactors) {
            pushCandidate(0, 0);
            return;
        }

        const size_t countA = std::min(topK, sortedActionsA.size());
        const size_t countB = std::min(topK, sortedActionsB.size());
        std::vector<JointActionCandidate> scoredCandidates;
        std::vector<float> logits;
        scoredCandidates.reserve(countA * countB);
        logits.reserve(countA * countB);
        float originalMass = 0.0f;
        for (size_t indexA = 0; indexA < countA; ++indexA) {
            for (size_t indexB = 0; indexB < countB; ++indexB) {
                JointActionCandidate candidate(
                    sortedActionsA[indexA], sortedPriorsA[indexA], indexA,
                    sortedActionsB[indexB], sortedPriorsB[indexB], indexB,
                    rules, sortedCapturesA[indexA] != 0,
                    sortedCapturesB[indexB] != 0);
                if (candidate.jointPrior < 0.0f) {
                    continue;
                }
                float compatibility = 0.0f;
                for (size_t factor = 0; factor < jointFactorRank; ++factor) {
                    compatibility +=
                        sortedJointFactorsA[indexA * jointFactorRank + factor]
                        * sortedJointFactorsB[indexB * jointFactorRank + factor];
                }
                compatibility /= std::sqrt(static_cast<float>(jointFactorRank));
                if (!std::isfinite(compatibility)) {
                    compatibility = 0.0f;
                }
                compatibility = std::clamp(compatibility, -8.0f, 8.0f);
                logits.push_back(
                    std::log(std::max(candidate.jointPrior, 1.0e-30f))
                    + residualScale * compatibility);
                originalMass += candidate.jointPrior;
                jointPoolKeys.emplace(indexA, indexB);
                scoredCandidates.push_back(candidate);
            }
        }
        if (scoredCandidates.empty()) {
            pushCandidate(0, 0);
            return;
        }

        const float maximum = *std::max_element(logits.begin(), logits.end());
        float normalizer = 0.0f;
        for (float& logit : logits) {
            logit = std::exp(logit - maximum);
            normalizer += logit;
        }
        for (size_t index = 0; index < scoredCandidates.size(); ++index) {
            JointActionCandidate& candidate = scoredCandidates[index];
            candidate.jointPrior = originalMass * logits[index] / normalizer;
            candidate.expansionPriority = candidate.jointPrior;
            candidate.gumbelScore = std::log(
                std::max(candidate.jointPrior, 1.0e-30f));
            const auto generated = generatedIndices.find(
                std::make_pair(candidate.idxA, candidate.idxB));
            if (generated != generatedIndices.end()) {
                generatedCandidates[generated->second] = candidate;
            } else {
                jointPolicyCandidates.push_back(candidate);
            }
        }
        std::sort(
            jointPolicyCandidates.begin(), jointPolicyCandidates.end(),
            [](const JointActionCandidate& left,
               const JointActionCandidate& right) {
                return left.expansionPriority > right.expansionPriority;
            });

        pushCandidate(0, 0);
    }

public:
    JointCandidateGenerator() = default;

    void restoreFactorizedOrder() {
        for (size_t index = nextGumbelCandidate;
             index < gumbelCandidates.size(); ++index) {
            gumbelCandidates[index].expansionPriority =
                gumbelCandidates[index].jointPrior;
            gumbelCandidates[index].gumbelScore = std::log(
                std::max(gumbelCandidates[index].jointPrior, 1.0e-30f));
            heap.push(gumbelCandidates[index]);
        }
        gumbelCandidates.clear();
        nextGumbelCandidate = 0;
    }

    /**
     * @brief Initialize the generator with actions and priors from both boards.
     * @param actionsA Moves for board 0
     * @param actionsB Moves for board 1
     * @param priorsA Prior probabilities for board 0 moves
     * @param priorsB Prior probabilities for board 1 moves
     * @param hasTimeAdvantage If true, team is up on time and can sit when on turn
     * @param isAOnTurn True if it's this team's turn on board A
     * @param isBOnTurn True if it's this team's turn on board B
     * @param capturesA Per-move capture flags for board 0 (missing entries count as non-captures)
     * @param capturesB Per-move capture flags for board 1 (missing entries count as non-captures)
     */
    void initialize(const std::vector<Stockfish::Move>& actionsA,
                    const std::vector<Stockfish::Move>& actionsB,
                    const std::vector<float>& priorsA,
                    const std::vector<float>& priorsB,
                    bool hasTimeAdvantage,
                    bool isAOnTurn,
                    bool isBOnTurn,
                    const std::vector<uint8_t>& capturesA = {},
                    const std::vector<uint8_t>& capturesB = {},
                    const std::vector<float>& jointFactorsA = {},
                    const std::vector<float>& jointFactorsB = {},
                    size_t jointFactorRank = 0,
                    size_t jointPolicyTopK = 8,
                    float jointResidualScale = 1.0f) {
        // Clear previous state
        sortedActionsA.clear();
        sortedActionsB.clear();
        sortedPriorsA.clear();
        sortedPriorsB.clear();
        sortedCapturesA.clear();
        sortedCapturesB.clear();
        sortedJointFactorsA.clear();
        sortedJointFactorsB.clear();
        while (!heap.empty()) heap.pop();
        visited.clear();
        generatedCandidates.clear();
        jointPoolKeys.clear();
        generatedCandidateKeys.clear();
        jointPolicyCandidates.clear();
        nextJointPolicyCandidate = 0;
        gumbelCandidates.clear();
        nextGumbelCandidate = 0;
        this->jointFactorRank = 0;
        
        // Use explicit on-turn status (not inferred from action count,
        // since a board can be on-turn but stalemated with no legal moves)
        rules = JointActionRules();
        rules.boardAOnTurn = isAOnTurn;
        rules.boardBOnTurn = isBOnTurn;
        rules.teamHasTimeAdvantage = hasTimeAdvantage;
        auto hasRealMove = [](const std::vector<Stockfish::Move>& actions) {
            return std::any_of(actions.begin(), actions.end(),
                               [](Stockfish::Move move) { return move != Stockfish::MOVE_NONE; });
        };
        rules.boardACanMove = isAOnTurn && hasRealMove(actionsA);
        rules.boardBCanMove = isBOnTurn && hasRealMove(actionsB);
        
        if (actionsA.empty() || actionsB.empty()) {
            return;
        }
        
        // Get indices sorted by prior (descending) for board A
        std::vector<size_t> indicesA(actionsA.size());
        std::iota(indicesA.begin(), indicesA.end(), 0);
        std::sort(indicesA.begin(), indicesA.end(),
                  [&priorsA](size_t i, size_t j) { return priorsA[i] > priorsA[j]; });
        
        // Get indices sorted by prior (descending) for board B
        std::vector<size_t> indicesB(actionsB.size());
        std::iota(indicesB.begin(), indicesB.end(), 0);
        std::sort(indicesB.begin(), indicesB.end(),
                  [&priorsB](size_t i, size_t j) { return priorsB[i] > priorsB[j]; });
        
        auto captureAt = [](const std::vector<uint8_t>& captures, size_t index) -> uint8_t {
            return index < captures.size() ? captures[index] : uint8_t{0};
        };

        // Build sorted arrays
        sortedActionsA.reserve(actionsA.size());
        sortedPriorsA.reserve(priorsA.size());
        sortedCapturesA.reserve(actionsA.size());
        for (size_t idx : indicesA) {
            sortedActionsA.push_back(actionsA[idx]);
            sortedPriorsA.push_back(priorsA[idx]);
            sortedCapturesA.push_back(captureAt(capturesA, idx));
            if (jointFactorRank > 0 &&
                jointFactorsA.size() == actionsA.size() * jointFactorRank) {
                sortedJointFactorsA.insert(
                    sortedJointFactorsA.end(),
                    jointFactorsA.begin() + idx * jointFactorRank,
                    jointFactorsA.begin() + (idx + 1) * jointFactorRank);
            }
        }
        
        sortedActionsB.reserve(actionsB.size());
        sortedPriorsB.reserve(priorsB.size());
        sortedCapturesB.reserve(actionsB.size());
        for (size_t idx : indicesB) {
            sortedActionsB.push_back(actionsB[idx]);
            sortedPriorsB.push_back(priorsB[idx]);
            sortedCapturesB.push_back(captureAt(capturesB, idx));
            if (jointFactorRank > 0 &&
                jointFactorsB.size() == actionsB.size() * jointFactorRank) {
                sortedJointFactorsB.insert(
                    sortedJointFactorsB.end(),
                    jointFactorsB.begin() + idx * jointFactorRank,
                    jointFactorsB.begin() + (idx + 1) * jointFactorRank);
            }
        }
        
        this->jointFactorRank = jointFactorRank;
        rebuildCandidateFrontier(jointPolicyTopK, jointResidualScale);
    }

    /**
     * Rebuild the unexpanded policy ordering when a retained node becomes the
     * root. Expanded action indices remain stable; their priors are rescored in
     * place and returned for the owning Node's edge arrays.
     */
    std::vector<float> reprepareJointPolicyPool(
        size_t topK, float residualScale) {
        rebuildCandidateFrontier(topK, residualScale);
        std::vector<float> generatedPriors;
        generatedPriors.reserve(generatedCandidates.size());
        for (const JointActionCandidate& candidate : generatedCandidates) {
            generatedPriors.push_back(candidate.jointPrior);
        }
        return generatedPriors;
    }

    /**
     * @brief Perturb a large policy-ordered root pool with independent Gumbels.
     *
     * Previously prepared but unexpanded candidates are included when a reused
     * node becomes the root of a new search, so no legal action is discarded.
     */
    void prepareGumbelPool(size_t poolSize, uint64_t seed) {
        if (poolSize == 0) {
            return;
        }

        std::vector<JointActionCandidate> pool;
        pool.reserve(poolSize);
        for (size_t index = nextGumbelCandidate;
             index < gumbelCandidates.size() && pool.size() < poolSize;
             ++index) {
            pool.push_back(gumbelCandidates[index]);
        }
        const size_t firstOverflow = nextGumbelCandidate + pool.size();
        nextGumbelCandidate = firstOverflow;
        restoreFactorizedOrder();
        gumbelCandidates.clear();
        nextGumbelCandidate = 0;

        while (pool.size() < poolSize &&
               (nextJointPolicyCandidate < jointPolicyCandidates.size() ||
                !heap.empty())) {
            pool.push_back(popBestUnperturbedCandidate());
        }

        std::mt19937_64 randomEngine(seed);
        std::uniform_real_distribution<double> uniform(
            std::nextafter(0.0, 1.0), std::nextafter(1.0, 0.0));
        for (JointActionCandidate& candidate : pool) {
            const double sample = uniform(randomEngine);
            const float gumbel = static_cast<float>(
                -std::log(-std::log(sample)));
            candidate.gumbelScore = std::log(
                std::max(candidate.jointPrior, 1.0e-30f)) + gumbel;
        }
        std::sort(pool.begin(), pool.end(),
                  [](const JointActionCandidate& lhs,
                     const JointActionCandidate& rhs) {
                      return lhs.gumbelScore > rhs.gumbelScore;
                  });
        gumbelCandidates = std::move(pool);
    }

    /**
     * @brief Check if there are more candidates to generate.
     */
    bool hasNext() const {
        return nextGumbelCandidate < gumbelCandidates.size()
            || nextJointPolicyCandidate < jointPolicyCandidates.size()
            || !heap.empty();
    }

    size_t pendingGumbelCount() const {
        return gumbelCandidates.size() - nextGumbelCandidate;
    }

    /**
     * @brief Peek at the next best candidate without consuming it.
     * 
     * Returns the next candidate that would be returned by getNext(),
     * but doesn't modify any state. Gumbel candidates retain root precedence;
     * otherwise the learned and factorized frontiers compete by priority.
     */
    JointActionCandidate peekNext() const {
        if (nextGumbelCandidate < gumbelCandidates.size()) {
            return gumbelCandidates[nextGumbelCandidate];
        }
        const bool hasPolicyCandidate =
            nextJointPolicyCandidate < jointPolicyCandidates.size();
        if (!hasPolicyCandidate && heap.empty()) {
            return JointActionCandidate();
        }

        if (hasPolicyCandidate &&
            (heap.empty() ||
             jointPolicyCandidates[nextJointPolicyCandidate].expansionPriority
                 >= heap.top().expansionPriority)) {
            return jointPolicyCandidates[nextJointPolicyCandidate];
        }
        
        // Invalid factorized candidates never enter the heap.
        return heap.top();
    }

    /**
     * @brief Get the next best joint action candidate.
     * 
     * Pops the current best from the heap and pushes the next candidates.
     * Invalid candidates are never pushed to the heap, so no filtering needed here.
     */
    JointActionCandidate getNext() {
        if (!hasNext()) {
            return JointActionCandidate();
        }

        JointActionCandidate best;
        if (nextGumbelCandidate < gumbelCandidates.size()) {
            best = gumbelCandidates[nextGumbelCandidate++];
        } else {
            best = popBestUnperturbedCandidate();
        }
        
        // Cache for random access
        generatedCandidates.push_back(best);
        generatedCandidateKeys.emplace(best.idxA, best.idxB);
        
        return best;
    }

    /**
     * @brief Get an already-generated candidate by index.
     * Returns empty candidate if index is out of bounds.
     */
    const JointActionCandidate& getGenerated(size_t idx) const {
        static const JointActionCandidate empty;
        if (idx >= generatedCandidates.size()) {
            return empty;
        }
        return generatedCandidates[idx];
    }

    /**
     * @brief Get the number of candidates generated so far.
     */
    size_t generatedCount() const {
        return generatedCandidates.size();
    }

    /**
     * @brief Get total possible candidates (N * M).
     */
    size_t totalPossible() const {
        return sortedActionsA.size() * sortedActionsB.size();
    }

    /**
     * @brief Check if generator is empty (no moves on one or both boards).
     */
    bool isEmpty() const {
        return sortedActionsA.empty() || sortedActionsB.empty();
    }
};
