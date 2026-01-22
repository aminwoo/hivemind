#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <set>
#include <functional>
#include <numeric>
#include "Fairy-Stockfish/src/types.h"

// Progressive Widening hyperparameters
constexpr float PW_COEFFICIENT = 1.5f;     
constexpr float PW_EXPONENT = 0.5f;   

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
    size_t idxA;                   // Index in sorted actionsA
    size_t idxB;                   // Index in sorted actionsB

    JointActionCandidate() 
        : moveA(Stockfish::MOVE_NONE), moveB(Stockfish::MOVE_NONE),
          priorA(0.0f), priorB(0.0f), jointPrior(0.0f),
          idxA(0), idxB(0) {}

    JointActionCandidate(Stockfish::Move mA, float pA, size_t iA,
                         Stockfish::Move mB, float pB, size_t iB,
                         bool boardAOnTurn = false,
                         bool boardBOnTurn = false,
                         bool teamHasTimeAdvantage = false)
        : moveA(mA), moveB(mB),
          priorA(pA), priorB(pB),
          idxA(iA), idxB(iB) {
        // Sitting rules:
        // 1. Both boards cannot sit simultaneously if both are on turn
        // 2. If team has time advantage: a board may sit even if on turn
        // 3. If team is behind on time: a board may only sit if NOT on turn
        bool isInvalidSit = false;
        bool bothSitting = (mA == Stockfish::MOVE_NULL && mB == Stockfish::MOVE_NULL);
        
        // Rule 1: Both boards cannot sit simultaneously if both are on turn
        if (bothSitting && boardAOnTurn && boardBOnTurn) {
            isInvalidSit = true;
        }
        
        // Rule 2 & 3: Check individual board sitting validity
        if (!teamHasTimeAdvantage) {
            // Behind on time: can only sit if NOT on turn
            if (boardAOnTurn && mA == Stockfish::MOVE_NULL) isInvalidSit = true;
            if (boardBOnTurn && mB == Stockfish::MOVE_NULL) isInvalidSit = true;
        }
        
        jointPrior = isInvalidSit ? 0.0f : pA * pB;
    }

    // For max-heap comparison (we want highest jointPrior first)
    bool operator<(const JointActionCandidate& other) const {
        return jointPrior < other.jointPrior;
    }
};

/**
 * @brief Lazy priority queue generator for joint action candidates.
 * 
 * Instead of generating all N*M pairs upfront, this class uses a max-heap
 * to lazily generate pairs in order of decreasing joint prior P_A * P_B.
 * 
 * Algorithm:
 * 1. Sort moves from each board by prior (descending)
 * 2. Start with (0,0) - the best pair from both boards
 * 3. When popping (i,j), push (i+1,j) and (i,j+1) if not already visited
 * 4. This guarantees pairs are generated in order of joint prior
 */
class JointCandidateGenerator {
private:
    // Sorted actions and priors for each board
    std::vector<Stockfish::Move> sortedActionsA;
    std::vector<Stockfish::Move> sortedActionsB;
    std::vector<float> sortedPriorsA;
    std::vector<float> sortedPriorsB;
    
    // Max-heap for lazy generation
    std::priority_queue<JointActionCandidate> heap;
    
    // Track visited (i,j) pairs to avoid duplicates
    std::set<std::pair<size_t, size_t>> visited;
    
    // Cache of already-generated candidates (for random access)
    std::vector<JointActionCandidate> generatedCandidates;
    
    // Track if each board is on turn (has real actions, not just null move)
    bool boardAOnTurn = false;
    bool boardBOnTurn = false;
    
    // Track if team has time advantage (allows sitting when on turn)
    bool teamHasTimeAdvantage = false;

    void pushCandidate(size_t idxA, size_t idxB) {
        if (idxA >= sortedActionsA.size() || idxB >= sortedActionsB.size()) {
            return;
        }
        
        auto key = std::make_pair(idxA, idxB);
        if (visited.find(key) != visited.end()) {
            return;
        }
        visited.insert(key);
        
        heap.emplace(
            sortedActionsA[idxA], sortedPriorsA[idxA], idxA,
            sortedActionsB[idxB], sortedPriorsB[idxB], idxB,
            boardAOnTurn,
            boardBOnTurn,
            teamHasTimeAdvantage
        );
    }

public:
    JointCandidateGenerator() = default;

    /**
     * @brief Initialize the generator with actions and priors from both boards.
     * @param actionsA Moves for board 0
     * @param actionsB Moves for board 1
     * @param priorsA Prior probabilities for board 0 moves
     * @param priorsB Prior probabilities for board 1 moves
     * @param hasTimeAdvantage If true, team is up on time and can sit when on turn
     */
    void initialize(const std::vector<Stockfish::Move>& actionsA,
                    const std::vector<Stockfish::Move>& actionsB,
                    const std::vector<float>& priorsA,
                    const std::vector<float>& priorsB,
                    bool hasTimeAdvantage = false) {
        // Clear previous state
        sortedActionsA.clear();
        sortedActionsB.clear();
        sortedPriorsA.clear();
        sortedPriorsB.clear();
        while (!heap.empty()) heap.pop();
        visited.clear();
        generatedCandidates.clear();
        
        // A board is "on turn" if it has more than just the null move
        boardAOnTurn = (actionsA.size() > 1);
        boardBOnTurn = (actionsB.size() > 1);
        teamHasTimeAdvantage = hasTimeAdvantage;
        
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
        
        // Build sorted arrays
        sortedActionsA.reserve(actionsA.size());
        sortedPriorsA.reserve(priorsA.size());
        for (size_t idx : indicesA) {
            sortedActionsA.push_back(actionsA[idx]);
            sortedPriorsA.push_back(priorsA[idx]);
        }
        
        sortedActionsB.reserve(actionsB.size());
        sortedPriorsB.reserve(priorsB.size());
        for (size_t idx : indicesB) {
            sortedActionsB.push_back(actionsB[idx]);
            sortedPriorsB.push_back(priorsB[idx]);
        }
        
        // Initialize heap with the best pair (0, 0)
        pushCandidate(0, 0);
    }

    /**
     * @brief Check if there are more candidates to generate.
     */
    bool hasNext() const {
        return !heap.empty();
    }

    /**
     * @brief Get the next best joint action candidate.
     * 
     * Pops the current best from the heap and pushes the next candidates.
     */
    JointActionCandidate getNext() {
        if (heap.empty()) {
            return JointActionCandidate();
        }
        
        JointActionCandidate best = heap.top();
        heap.pop();
        
        // Push adjacent candidates (i+1, j) and (i, j+1)
        pushCandidate(best.idxA + 1, best.idxB);
        pushCandidate(best.idxA, best.idxB + 1);
        
        // Cache for random access
        generatedCandidates.push_back(best);
        
        return best;
    }

    /**
     * @brief Get an already-generated candidate by index.
     */
    const JointActionCandidate& getGenerated(size_t idx) const {
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

/**
 * @brief Calculates the number of allowed children based on progressive widening.
 * 
 * Formula: m = ceil(PW_COEFFICIENT * n^PW_EXPONENT)
 * Where n is the visit count of the current node.
 * 
 * @param visitCount Current visit count of the node
 * @return int Number of children allowed to be expanded
 */
inline int get_allowed_children(int visitCount) {
    if (visitCount <= 0) return 1;
    return static_cast<int>(std::ceil(PW_COEFFICIENT * std::pow(static_cast<float>(visitCount), PW_EXPONENT)));
}
