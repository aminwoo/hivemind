#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>
#include <functional>
#include <numeric>
#include "Fairy-Stockfish/src/types.h"
#include "search_params.h"

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

        bool bothSitting = (mA == Stockfish::MOVE_NONE) && (mB == Stockfish::MOVE_NONE);
        bool bothOnTurn = boardAOnTurn && boardBOnTurn;
        bool isInvalidSit = bothSitting && (teamHasTimeAdvantage ? bothOnTurn : true);
        
        jointPrior = isInvalidSit ? -1.0f : pA * pB;
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
    
    // Track visited (i,j) pairs to avoid duplicates - O(1) lookup
    std::unordered_set<std::pair<size_t, size_t>, PairHash> visited;
    
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
        
        // Create candidate to check if it's valid
        JointActionCandidate candidate(
            sortedActionsA[idxA], sortedPriorsA[idxA], idxA,
            sortedActionsB[idxB], sortedPriorsB[idxB], idxB,
            boardAOnTurn,
            boardBOnTurn,
            teamHasTimeAdvantage
        );
        
        // Only push valid candidates (jointPrior != -1.0f)
        if (candidate.jointPrior >= 0.0f) {
            heap.push(candidate);
        }
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
        // If (0, 0) is invalid (e.g., double pass), also try (1, 0) and (0, 1)
        pushCandidate(0, 0);
        
        // If (0, 0) was invalid and not pushed, ensure we have alternatives
        if (heap.empty()) {
            pushCandidate(1, 0);
            pushCandidate(0, 1);
        }
    }

    /**
     * @brief Check if there are more candidates to generate.
     */
    bool hasNext() const {
        return !heap.empty();
    }

    /**
     * @brief Peek at the next best candidate without consuming it.
     * 
     * Returns the next candidate that would be returned by getNext(),
     * but doesn't modify any state. This is a true peek operation.
     * Note: For efficiency, this only checks the top of heap. If the top
     * has zero jointPrior (invalid), returns empty candidate - the caller
     * should handle this by falling through to getNext() which properly skips.
     */
    JointActionCandidate peekNext() const {
        if (heap.empty()) {
            return JointActionCandidate();
        }
        
        // Return the top candidate - caller should check jointPrior
        // We can't skip invalid candidates here without modifying state
        return heap.top();
    }

    /**
     * @brief Get the next best joint action candidate.
     * 
     * Pops the current best from the heap and pushes the next candidates.
     * Invalid candidates are never pushed to the heap, so no filtering needed here.
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
