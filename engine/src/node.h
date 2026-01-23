#pragma once

#include <cmath>
#include <memory>
#include <mutex>
#include <vector>

#include "board.h"
#include "constants.h"
#include "joint_action.h"
#include "search_params.h"
#include "utils.h"
#include "Fairy-Stockfish/src/types.h"

class Node {
private:
    // Mutex for thread-safe access to node state
    mutable std::mutex nodeMutex;
    
    // Children indexed by expansion order
    std::vector<std::shared_ptr<Node>> children;
    
    // Joint action support for Progressive Widening with lazy priority queue
    JointCandidateGenerator candidateGenerator;
    int expandedCount = 0;
    
    // MCTS statistics
    std::vector<float> qValues;
    std::vector<float> childValueSum;
    std::vector<float> childPriors;
    std::vector<int> childVisits;
    std::vector<int> virtualLoss;  // Virtual visits for batch MCTS
    int bestChildIdx = -1;

    float valueSum = -1.0f;
    int m_depth = 0;
    int m_visits = 0;
    bool m_is_expanded = false;

    Stockfish::Color teamToPlay;

public:
    Node(Stockfish::Color teamToPlay) : teamToPlay(teamToPlay) {}
    ~Node() = default;

    // Lock the node for exclusive access
    std::unique_lock<std::mutex> lock() {
        return std::unique_lock<std::mutex>(nodeMutex);
    }

    void update(size_t childIdx, float value) {
        std::lock_guard<std::mutex> guard(nodeMutex);
        childValueSum[childIdx] += value;
        childVisits[childIdx]++;
        qValues[childIdx] = childValueSum[childIdx] / (1.0f + childVisits[childIdx]);
        valueSum += value;
        m_visits++;
    }

    void update_terminal(float value) {
        std::lock_guard<std::mutex> guard(nodeMutex);
        valueSum += value;
        m_visits++;
    }

    void set_depth(int value) {
        m_depth = value;
    }

    int get_depth() {
        return m_depth;
    }

    int get_best_child_idx() {
        return bestChildIdx;
    }

    void set_best_child_idx(int value) {
        bestChildIdx = value;
    }

    bool is_terminal() const;

    /**
     * @brief Initializes the lazy priority queue generator for joint actions.
     * Thread-safe: uses internal locking.
     * 
     * Uses a max-heap to lazily generate pairs in order of decreasing P_A * P_B.
     * This considers ALL moves from both boards without arbitrary top-K filtering.
     * @param teamHasTimeAdvantage If true, team is up on time and can sit when on turn
     */
    void init_joint_generator(const std::vector<Stockfish::Move>& actionsA,
                              const std::vector<Stockfish::Move>& actionsB,
                              const std::vector<float>& priorsA,
                              const std::vector<float>& priorsB,
                              bool teamHasTimeAdvantage = false) {
        std::lock_guard<std::mutex> guard(nodeMutex);
        candidateGenerator.initialize(actionsA, actionsB, priorsA, priorsB, teamHasTimeAdvantage);
        expandedCount = 0;
    }

    /**
     * @brief Returns the number of currently expanded children.
     */
    int get_expanded_count() const {
        return expandedCount;
    }

    /**
     * @brief Returns the total number of possible joint actions (N * M).
     */
    size_t get_candidate_count() const {
        return candidateGenerator.totalPossible();
    }

    /**
     * @brief Checks if we should expand a new child based on progressive widening.
     * 
     * Progressive widening formula: m = ceil(C_PW * n^KAPPA)
     * Returns true if expandedCount < allowedChildren AND generator has more candidates.
     */
    bool should_expand_new_child() {
        std::lock_guard<std::mutex> guard(nodeMutex);
        int allowedChildren = SearchParams::get_allowed_children(m_visits);
        return expandedCount < allowedChildren && candidateGenerator.hasNext();
    }

    /**
     * @brief Expands the next joint action candidate using a Parent-Relative FPU strategy.
     * Thread-safe: uses internal locking.
     */
    std::shared_ptr<Node> expand_next_joint_child() {
        std::lock_guard<std::mutex> guard(nodeMutex);
        if (!candidateGenerator.hasNext()) {
            return nullptr;
        }

        JointActionCandidate candidate = candidateGenerator.getNext();
        
        auto child = std::make_shared<Node>(~teamToPlay);
        child->set_depth(m_depth + 1);
        
        float parentQ = (m_visits > 0) ? (valueSum / m_visits) : 0.0f;

        float fpuValue = parentQ - SearchParams::FPU_REDUCTION;

        fpuValue = std::max(-1.0f, std::min(1.0f, fpuValue));

        child->set_value(fpuValue);
        
        childValueSum.push_back(fpuValue);
        childPriors.push_back(candidate.jointPrior);
        childVisits.push_back(0);
        virtualLoss.push_back(0);
        children.push_back(child);
        qValues.push_back(fpuValue);
        
        expandedCount++;
        return child;
    }

    /**
     * @brief Atomically initializes and expands the first child if not already expanded.
     * Thread-safe: Ensures only one thread initializes the node.
     * @return true if this thread performed the initialization, false if already expanded.
     */
    bool try_init_and_expand(const std::vector<Stockfish::Move>& actionsA,
                             const std::vector<Stockfish::Move>& actionsB,
                             const std::vector<float>& priorsA,
                             const std::vector<float>& priorsB,
                             bool teamHasTimeAdvantage = false) {
        std::lock_guard<std::mutex> guard(nodeMutex);
        
        // Already expanded by another thread
        if (m_is_expanded) {
            return false;
        }
        
        // Initialize generator
        candidateGenerator.initialize(actionsA, actionsB, priorsA, priorsB, teamHasTimeAdvantage);
        expandedCount = 0;
        
        // Try to expand the first child
        if (candidateGenerator.hasNext()) {
            JointActionCandidate candidate = candidateGenerator.getNext();
            
            auto child = std::make_shared<Node>(~teamToPlay);
            child->set_depth(m_depth + 1);
            
            float parentQ = (m_visits > 0) ? (valueSum / m_visits) : 0.0f;
            float fpuValue = parentQ - SearchParams::FPU_REDUCTION;
            fpuValue = std::max(-1.0f, std::min(1.0f, fpuValue));
            
            child->set_value(fpuValue);
            
            childValueSum.push_back(fpuValue);
            childPriors.push_back(candidate.jointPrior);
            childVisits.push_back(0);
            virtualLoss.push_back(0);
            children.push_back(child);
            qValues.push_back(fpuValue);
            
            expandedCount++;
            bestChildIdx = 0;
            m_is_expanded = true;
            return true;
        }
        
        return false;
    }

    /**
     * @brief Gets the joint action for a specific child index (from generated cache).
     */
    JointActionCandidate get_joint_action(int childIdx) const {
        if (childIdx >= 0 && static_cast<size_t>(childIdx) < candidateGenerator.generatedCount()) {
            return candidateGenerator.getGenerated(childIdx);
        }
        return JointActionCandidate();
    }

    /**
     * @brief Checks if the joint candidate generator has candidates.
     */
    bool has_joint_candidates() const {
        return !candidateGenerator.isEmpty();
    }

    /**
     * @brief Selects the best child using PUCT among expanded children only.
     * 
     * This is used during the selection phase of MCTS with progressive widening.
     * Only considers children that have already been expanded.
     */
    std::shared_ptr<Node> get_best_expanded_child();

    /**
     * @brief Selects the best child and returns both the child pointer and index.
     * Thread-safe: Returns the child index atomically with the selection.
     * @return pair of (child pointer, child index), or (nullptr, -1) if no children
     */
    std::pair<std::shared_ptr<Node>, int> get_best_expanded_child_with_idx();

    std::vector<std::shared_ptr<Node>> get_children() {
        return children;
    }

    bool is_expanded() {
        std::lock_guard<std::mutex> guard(nodeMutex);
        return m_is_expanded;
    }

    void set_is_expanded(bool value) {
        std::lock_guard<std::mutex> guard(nodeMutex);
        m_is_expanded = value;
    }

    void set_value(float value) {
        valueSum = value;
    }

    Stockfish::Color get_team_to_play() {
        return teamToPlay;
    }

    Stockfish::Color get_child_team_to_play() {
        return children[bestChildIdx]->get_team_to_play();
    }

    void set_team_to_play(Stockfish::Color value) {
        teamToPlay = value;
    }

    int get_visits() {
        return m_visits;
    }

    void increment_visits() {
        m_visits++;
    }

    void decrement_visits() {
        m_visits--;
    }

    // Virtual loss methods for batch MCTS (thread-safe)
    void apply_virtual_loss(int childIdx, int amount = 1) {
        std::lock_guard<std::mutex> guard(nodeMutex);
        if (childIdx >= 0 && static_cast<size_t>(childIdx) < virtualLoss.size()) {
            virtualLoss[childIdx] += amount;
        }
    }

    void remove_virtual_loss(int childIdx, int amount = 1) {
        std::lock_guard<std::mutex> guard(nodeMutex);
        if (childIdx >= 0 && static_cast<size_t>(childIdx) < virtualLoss.size()) {
            virtualLoss[childIdx] -= amount;
        }
    }

    int get_virtual_loss(int childIdx) const {
        if (childIdx >= 0 && static_cast<size_t>(childIdx) < virtualLoss.size()) {
            return virtualLoss[childIdx];
        }
        return 0;
    }

    float get_value_sum() {
        return valueSum;
    }

    float Q() {
        return valueSum / (1.0f + m_visits);
    }
};
