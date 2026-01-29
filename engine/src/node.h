#pragma once

#include <atomic>
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
    
    // MCGS (Monte Carlo Graph Search) support
    uint64_t positionHash = 0;  // Zobrist hash for transposition detection
    std::atomic<bool> isTransposition{false};  // True if this node was found via transposition
    std::atomic<int> inFlightCount{0};  // Number of in-flight evaluations (for MCGS backup)

public:
    Node(Stockfish::Color teamToPlay) : teamToPlay(teamToPlay) {}
    Node(Stockfish::Color teamToPlay, uint64_t hash) 
        : teamToPlay(teamToPlay), positionHash(hash) {}
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
     * @brief Returns the number of generated candidates (should equal children.size()).
     */
    size_t get_num_generated() const {
        return candidateGenerator.generatedCount();
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
     * @brief Peek at the next joint action without consuming it.
     * Thread-safe: uses internal locking.
     * @return The next action that would be expanded, or empty if none available.
     */
    JointActionCandidate peek_next_joint_action() {
        std::lock_guard<std::mutex> guard(nodeMutex);
        return candidateGenerator.peekNext();
    }

    /**
     * @brief Expands the next joint action candidate using a Parent-Relative FPU strategy.
     * Thread-safe: uses internal locking.
     * 
     * MCGS Enhancement: Can accept an existing node from the transposition table.
     * If existingNode is provided, it will be used instead of creating a new node.
     * 
     * @param existingNode Optional node from transposition table to reuse
     * @param positionHash Hash of the resulting position for MCGS
     * @param outAction Output parameter for the action that was actually expanded
     * @return The child node (new or existing)
     */
    std::shared_ptr<Node> expand_next_joint_child(std::shared_ptr<Node> existingNode,
                                                   uint64_t positionHash,
                                                   JointActionCandidate& outAction) {
        std::lock_guard<std::mutex> guard(nodeMutex);
        if (!candidateGenerator.hasNext()) {
            return nullptr;
        }

        JointActionCandidate candidate = candidateGenerator.getNext();
        
        outAction = candidate;  // Return the actual action used
        
        std::shared_ptr<Node> child;
        float childQ;
        
        if (existingNode) {
            // MCGS: Reuse existing node from transposition table
            child = existingNode;
            child->set_transposition(true);
            childQ = existingNode->Q();
        } else {
            // Create new node
            child = std::make_shared<Node>(~teamToPlay, positionHash);
            child->set_depth(m_depth + 1);
            
            float parentQ = (m_visits > 0) ? (valueSum / m_visits) : 0.0f;
            float fpuValue = parentQ - SearchParams::FPU_REDUCTION;
            fpuValue = std::max(-1.0f, std::min(1.0f, fpuValue));
            child->set_value(fpuValue);
            childQ = fpuValue;
        }
        
        childValueSum.push_back(childQ);
        childPriors.push_back(candidate.jointPrior);
        childVisits.push_back(existingNode ? existingNode->get_visits() : 0);
        virtualLoss.push_back(0);
        children.push_back(child);
        qValues.push_back(childQ);
        
        expandedCount++;
        return child;
    }
    
    /**
     * @brief Simplified expand without MCGS (for backwards compatibility).
     */
    std::shared_ptr<Node> expand_next_joint_child() {
        JointActionCandidate unused;
        return expand_next_joint_child(nullptr, 0, unused);
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
        size_t genCount = candidateGenerator.generatedCount();
        if (childIdx >= 0 && static_cast<size_t>(childIdx) < genCount) {
            return candidateGenerator.getGenerated(childIdx);
        }
        std::cerr << "ERROR in get_joint_action: childIdx=" << childIdx 
                  << " out of bounds (generatedCount=" << genCount << ")" << std::endl;
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

    const std::vector<float>& get_child_value_sums() const {
        return childValueSum;
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
    
    /**
     * @brief Apply Dirichlet noise to child priors for exploration.
     * @param noise Dirichlet noise vector (same size as children)
     * @param epsilon Mixing factor (prior = (1 - epsilon) * prior + epsilon * noise)
     */
    void apply_dirichlet_noise(const std::vector<float>& noise, float epsilon) {
        std::lock_guard<std::mutex> guard(nodeMutex);
        if (noise.size() != childPriors.size()) return;
        
        for (size_t i = 0; i < childPriors.size(); ++i) {
            childPriors[i] = (1.0f - epsilon) * childPriors[i] + epsilon * noise[i];
        }
    }
    
    /**
     * @brief Get the number of children that have been expanded.
     */
    size_t get_num_children() const {
        return children.size();
    }
    
    /**
     * @brief Get the visit counts for all expanded children.
     * Used for extracting MCTS policy distributions.
     */
    std::vector<int> get_child_visits() const {
        std::lock_guard<std::mutex> guard(nodeMutex);
        return childVisits;
    }
    
    /**
     * @brief Get the joint action and visit count for each child.
     * Returns vector of (JointActionCandidate, visit_count) pairs.
     * Used for extracting MCTS policy distributions for training.
     */
    std::vector<std::pair<JointActionCandidate, int>> get_child_action_visits() const {
        std::lock_guard<std::mutex> guard(nodeMutex);
        std::vector<std::pair<JointActionCandidate, int>> result;
        result.reserve(children.size());
        
        for (size_t i = 0; i < children.size(); ++i) {
            JointActionCandidate action = candidateGenerator.getGenerated(i);
            int visits = (i < childVisits.size()) ? childVisits[i] : 0;
            result.emplace_back(action, visits);
        }
        
        return result;
    }
    
    /**
     * @brief Get the position hash for this node.
     * Used for transposition table lookups.
     */
    uint64_t get_hash() const {
        return positionHash;
    }
    
    /**
     * @brief Set the position hash for this node.
     */
    void set_hash(uint64_t hash) {
        positionHash = hash;
    }
    
    /**
     * @brief Check if this node was reached via transposition.
     */
    bool is_transposition() const {
        return isTransposition.load(std::memory_order_relaxed);
    }
    
    /**
     * @brief Mark this node as reached via transposition.
     */
    void set_transposition(bool value) {
        isTransposition.store(value, std::memory_order_relaxed);
    }
    
    /**
     * @brief Increment in-flight evaluation count (for batched MCGS).
     * Used to track pending backups through this node.
     */
    void increment_in_flight() {
        inFlightCount.fetch_add(1, std::memory_order_relaxed);
    }
    
    /**
     * @brief Decrement in-flight evaluation count.
     */
    void decrement_in_flight() {
        inFlightCount.fetch_sub(1, std::memory_order_relaxed);
    }
    
    /**
     * @brief Get current in-flight evaluation count.
     */
    int get_in_flight() const {
        return inFlightCount.load(std::memory_order_relaxed);
    }
    
    /**
     * @brief Atomically try to set this node as an existing child, returning the child index.
     * Used for MCGS when the same position is reached through different paths.
     * 
     * @param existingNode The node already in the transposition table
     * @return The index of the child that was added, or -1 if already exists
     */
    int try_add_transposition_child(std::shared_ptr<Node> existingNode, float prior) {
        std::lock_guard<std::mutex> guard(nodeMutex);
        
        // Check if this child already exists (same hash)
        for (size_t i = 0; i < children.size(); i++) {
            if (children[i] && children[i]->get_hash() == existingNode->get_hash()) {
                return static_cast<int>(i);  // Already exists
            }
        }
        
        // Add as new child
        float parentQ = (m_visits > 0) ? (valueSum / m_visits) : 0.0f;
        float fpuValue = parentQ - SearchParams::FPU_REDUCTION;
        fpuValue = std::max(-1.0f, std::min(1.0f, fpuValue));
        
        childValueSum.push_back(fpuValue);
        childPriors.push_back(prior);
        childVisits.push_back(existingNode->get_visits());  // Use existing visits
        virtualLoss.push_back(0);
        children.push_back(existingNode);
        qValues.push_back(existingNode->Q());  // Use existing Q value
        
        int idx = static_cast<int>(children.size() - 1);
        expandedCount++;
        
        return idx;
    }
    
    /**
     * @brief Get a child by position hash.
     * @return Pair of (child pointer, child index), or (nullptr, -1) if not found
     */
    std::pair<std::shared_ptr<Node>, int> get_child_by_hash(uint64_t hash) {
        std::lock_guard<std::mutex> guard(nodeMutex);
        for (size_t i = 0; i < children.size(); i++) {
            if (children[i] && children[i]->get_hash() == hash) {
                return {children[i], static_cast<int>(i)};
            }
        }
        return {nullptr, -1};
    }
};
