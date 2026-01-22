#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include "board.h"
#include "constants.h"
#include "joint_action.h"
#include "utils.h"
#include "Fairy-Stockfish/src/types.h"

class Node {
private:
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
    int bestChildIdx = -1;

    float valueSum = -1.0f;
    int m_depth = 0;
    int m_visits = 0;
    bool m_is_expanded = false;

    Stockfish::Color teamToPlay;

public:
    Node(Stockfish::Color teamToPlay) : teamToPlay(teamToPlay) {}
    ~Node() = default;


    void update(size_t childIdx, float value) {
        childValueSum[childIdx] += value;
        childVisits[childIdx]++;
        qValues[childIdx] = childValueSum[childIdx] / (1.0f + childVisits[childIdx]);
        valueSum += value;
        m_visits++;
    }

    void update_terminal(float value) {
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
        int allowedChildren = get_allowed_children(m_visits);
        return expandedCount < allowedChildren && candidateGenerator.hasNext();
    }

    /**
     * @brief Expands the next joint action candidate using a Parent-Relative FPU strategy.
     */
    std::shared_ptr<Node> expand_next_joint_child() {
        if (!candidateGenerator.hasNext()) {
            return nullptr;
        }

        JointActionCandidate candidate = candidateGenerator.getNext();
        
        auto child = std::make_shared<Node>(~teamToPlay);
        child->set_depth(m_depth + 1);
        
        float parentQ = (m_visits > 0) ? (valueSum / m_visits) : 0.0f;

        const float fpuReduction = 0.4f;
        float fpuValue = parentQ - fpuReduction;

        fpuValue = std::max(-1.0f, std::min(1.0f, fpuValue));

        child->set_value(fpuValue);
        
        childValueSum.push_back(fpuValue);
        childPriors.push_back(candidate.jointPrior);
        childVisits.push_back(0);
        children.push_back(child);
        qValues.push_back(fpuValue);
        
        expandedCount++;
        return child;
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

    std::vector<std::shared_ptr<Node>> get_children() {
        return children;
    }

    bool is_expanded() {
        return m_is_expanded;
    }

    void set_is_expanded(bool value) {
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

    float get_value_sum() {
        return valueSum;
    }

    float Q() {
        return valueSum / (1.0f + m_visits);
    }
};

float get_current_cput(float totalVisits);
