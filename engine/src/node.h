#pragma once

#include <atomic>
#include <cmath>
#include <memory>
#include <shared_mutex>
#include <vector>

#include "board.h"
#include "constants.h"
#include "joint_action.h"
#include "search_params.h"
#include "utils.h"
#include "Fairy-Stockfish/src/types.h"

/**
 * @brief Node type for MCTS Solver.
 * 
 * Tracks whether a node's game-theoretic value is proven:
 * - UNSOLVED: Not yet determined
 * - WIN: Proven winning position (opponent is mated)
 * - LOSS: Proven losing position (we are mated)
 * - DRAW: Proven draw (stalemate, repetition, etc.)
 */
enum class NodeType : uint8_t {
    UNSOLVED = 0,
    WIN = 1,
    LOSS = 2,
    DRAW = 3
};

class Node {
private:
    // Reader-writer mutex for thread-safe access to node state
    // Allows multiple concurrent readers, exclusive writers
    mutable std::shared_mutex nodeMutex;
    
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
    std::atomic<int> m_visits{0};  // Atomic for lock-free read/write
    bool m_is_expanded = false;

    Stockfish::Color teamToPlay;
    
    // MCGS (Monte Carlo Graph Search) support
    uint64_t positionHash = 0;  // Zobrist hash for transposition detection
    std::atomic<bool> isTransposition{false};  // True if this node was found via transposition
    std::atomic<int> inFlightCount{0};  // Number of in-flight evaluations (for MCGS backup)
    
    // MCTS Solver support
    NodeType nodeType = NodeType::UNSOLVED;  // Proven game-theoretic value
    std::vector<NodeType> childNodeTypes;    // Cached node types of children
    std::atomic<int> unsolvedChildCount{0};  // Number of unsolved children (for solver)
    int endInPly = 0;                        // Distance to terminal (for mate distance)

public:
    Node(Stockfish::Color teamToPlay) : teamToPlay(teamToPlay) {}
    Node(Stockfish::Color teamToPlay, uint64_t hash) 
        : teamToPlay(teamToPlay), positionHash(hash) {}
    ~Node() = default;

    // Lock the node for exclusive access
    std::unique_lock<std::shared_mutex> lock() {
        return std::unique_lock<std::shared_mutex>(nodeMutex);
    }

    /**
     * @brief Reserve capacity for child vectors to reduce reallocations.
     * Call this when you know the approximate number of children.
     */
    void reserve_children(size_t capacity) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        children.reserve(capacity);
        childValueSum.reserve(capacity);
        childPriors.reserve(capacity);
        childVisits.reserve(capacity);
        virtualLoss.reserve(capacity);
        qValues.reserve(capacity);
    }

    void update(size_t childIdx, float value) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        childVisits[childIdx]++;
        
        if (childVisits[childIdx] == 1) {
            // First visit: replace FPU initialization with actual value
            childValueSum[childIdx] = value;
            qValues[childIdx] = value;
        } else {
            // Subsequent visits: accumulate and average
            childValueSum[childIdx] += value;
            qValues[childIdx] = childValueSum[childIdx] / static_cast<float>(childVisits[childIdx]);
        }
        
        valueSum += value;
        m_visits.fetch_add(1, std::memory_order_relaxed);
    }

    void update_terminal(float value) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        valueSum += value;
        m_visits.fetch_add(1, std::memory_order_relaxed);
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
     * @param boardAOnTurn True if it's this team's turn on board A
     * @param boardBOnTurn True if it's this team's turn on board B
     */
    void init_joint_generator(const std::vector<Stockfish::Move>& actionsA,
                              const std::vector<Stockfish::Move>& actionsB,
                              const std::vector<float>& priorsA,
                              const std::vector<float>& priorsB,
                              bool teamHasTimeAdvantage,
                              bool boardAOnTurn,
                              bool boardBOnTurn) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        candidateGenerator.initialize(actionsA, actionsB, priorsA, priorsB, teamHasTimeAdvantage, boardAOnTurn, boardBOnTurn);
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
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return candidateGenerator.totalPossible();
    }

    /**
     * @brief Returns the number of generated candidates (should equal children.size()).
     */
    size_t get_num_generated() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return candidateGenerator.generatedCount();
    }

    /**
     * @brief Checks if we should expand a new child based on progressive widening.
     * 
     * Progressive widening formula: m = ceil(C_PW * n^KAPPA)
     * Returns true if expandedCount < allowedChildren AND generator has more candidates.
     */
    bool should_expand_new_child() {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        int allowedChildren = SearchParams::get_allowed_children(m_visits.load(std::memory_order_relaxed));
        return expandedCount < allowedChildren && candidateGenerator.hasNext();
    }

    /**
     * @brief Peek at the next joint action without consuming it.
     * Thread-safe: uses internal locking.
     * @return The next action that would be expanded, or empty if none available.
     */
    JointActionCandidate peek_next_joint_action() {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
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
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
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
            
            int visits = m_visits.load(std::memory_order_relaxed);
            float parentQ = (visits > 0) ? (valueSum / visits) : 0.0f;
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
                             bool teamHasTimeAdvantage,
                             bool boardAOnTurn,
                             bool boardBOnTurn) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        
        // Already expanded by another thread
        if (m_is_expanded) {
            return false;
        }
        
        // Initialize generator
        candidateGenerator.initialize(actionsA, actionsB, priorsA, priorsB, teamHasTimeAdvantage, boardAOnTurn, boardBOnTurn);
        expandedCount = 0;
        
        // Try to expand the first child
        if (candidateGenerator.hasNext()) {
            JointActionCandidate candidate = candidateGenerator.getNext();
        
            auto child = std::make_shared<Node>(~teamToPlay);
            child->set_depth(m_depth + 1);
            
            int visits = m_visits.load(std::memory_order_relaxed);
            float parentQ = (visits > 0) ? (valueSum / visits) : 0.0f;
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
     * Thread-safe: holds nodeMutex to ensure consistency with children vector.
     */
    JointActionCandidate get_joint_action(int childIdx) const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        size_t genCount = candidateGenerator.generatedCount();
        if (childIdx >= 0 && static_cast<size_t>(childIdx) < genCount) {
            return candidateGenerator.getGenerated(childIdx);
        }
        std::cerr << "ERROR in get_joint_action: childIdx=" << childIdx 
                  << " out of bounds (generatedCount=" << genCount 
                  << ", children.size=" << children.size() << ")" << std::endl;
        return JointActionCandidate();
    }

    /**
     * @brief Checks if the joint candidate generator has candidates.
     */
    bool has_joint_candidates() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
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

    std::vector<std::shared_ptr<Node>> get_children() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return children;
    }

    std::vector<float> get_child_value_sums() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return childValueSum;
    }

    bool is_expanded() {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return m_is_expanded;
    }

    void set_is_expanded(bool value) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
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
        return m_visits.load(std::memory_order_relaxed);
    }

    void increment_visits() {
        m_visits.fetch_add(1, std::memory_order_relaxed);
    }

    void decrement_visits() {
        m_visits.fetch_sub(1, std::memory_order_relaxed);
    }

    // Virtual loss methods for batch MCTS (thread-safe)
    void apply_virtual_loss(int childIdx, int amount = 1) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        if (childIdx >= 0 && static_cast<size_t>(childIdx) < virtualLoss.size()) {
            virtualLoss[childIdx] += amount;
        }
    }

    void remove_virtual_loss(int childIdx, int amount = 1) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
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

    float get_value_sum() const {
        return valueSum;
    }

    float Q() const {
        return valueSum / (1.0f + m_visits.load(std::memory_order_relaxed));
    }
    
    /**
     * @brief Apply Dirichlet noise to child priors for exploration.
     * @param noise Dirichlet noise vector (same size as children)
     * @param epsilon Mixing factor (prior = (1 - epsilon) * prior + epsilon * noise)
     */
    void apply_dirichlet_noise(const std::vector<float>& noise, float epsilon) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        if (noise.size() != childPriors.size()) return;
        
        for (size_t i = 0; i < childPriors.size(); ++i) {
            childPriors[i] = (1.0f - epsilon) * childPriors[i] + epsilon * noise[i];
        }
    }
    
    /**
     * @brief Get the number of children that have been expanded.
     */
    size_t get_num_children() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return children.size();
    }
    
    /**
     * @brief Get the visit counts for all expanded children.
     * Used for extracting MCTS policy distributions.
     */
    std::vector<int> get_child_visits() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return childVisits;
    }
    
    /**
     * @brief Get the joint action and visit count for each child.
     * Returns vector of (JointActionCandidate, visit_count) pairs.
     * Used for extracting MCTS policy distributions for training.
     * Note: Only includes children that have generated candidates (excludes transposition children).
     */
    std::vector<std::pair<JointActionCandidate, int>> get_child_action_visits() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        std::vector<std::pair<JointActionCandidate, int>> result;
        
        size_t genCount = candidateGenerator.generatedCount();
        size_t count = std::min(children.size(), genCount);
        result.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
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
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        
        // Check if this child already exists (same hash)
        for (size_t i = 0; i < children.size(); i++) {
            if (children[i] && children[i]->get_hash() == existingNode->get_hash()) {
                return static_cast<int>(i);  // Already exists
            }
        }
        
        // Add as new child
        int visits = m_visits.load(std::memory_order_relaxed);
        float parentQ = (visits > 0) ? (valueSum / visits) : 0.0f;
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
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        for (size_t i = 0; i < children.size(); i++) {
            if (children[i] && children[i]->get_hash() == hash) {
                return {children[i], static_cast<int>(i)};
            }
        }
        return {nullptr, -1};
    }
    
    // =========================================================================
    // MCTS Solver Methods
    // =========================================================================
    
    /**
     * @brief Get the node type (UNSOLVED, WIN, LOSS, DRAW).
     */
    NodeType get_node_type() const {
        return nodeType;
    }
    
    /**
     * @brief Set the node type (for solver propagation).
     */
    void set_node_type(NodeType type) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        nodeType = type;
    }
    
    /**
     * @brief Check if this node is solved (WIN, LOSS, or DRAW).
     */
    bool is_solved() const {
        return nodeType != NodeType::UNSOLVED;
    }
    
    /**
     * @brief Mark this node as a WIN (opponent is mated).
     */
    void mark_as_win(int ply = 0) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        nodeType = NodeType::WIN;
        valueSum = 1.0f * (m_visits.load(std::memory_order_relaxed) + 1);
        endInPly = ply;
    }
    
    /**
     * @brief Mark this node as a LOSS (we are mated).
     */
    void mark_as_loss(int ply = 0) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        nodeType = NodeType::LOSS;
        valueSum = -1.0f * (m_visits.load(std::memory_order_relaxed) + 1);
        endInPly = ply;
    }
    
    /**
     * @brief Mark this node as a DRAW.
     */
    void mark_as_draw() {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        nodeType = NodeType::DRAW;
        valueSum = -SearchParams::DRAW_CONTEMPT * (m_visits.load(std::memory_order_relaxed) + 1);
        endInPly = 0;
    }
    
    /**
     * @brief Get the ply distance to terminal.
     */
    int get_end_in_ply() const {
        return endInPly;
    }
    
    /**
     * @brief Initialize child node types array to match children count.
     * 
     * When new children are added via progressive widening, we need to
     * grow the childNodeTypes array and update the unsolved count.
     * We only add the NEW children to the unsolved count to preserve
     * the count of already-solved children.
     */
    void init_child_node_types() {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        if (childNodeTypes.size() < children.size()) {
            size_t oldSize = childNodeTypes.size();
            childNodeTypes.resize(children.size(), NodeType::UNSOLVED);
            // Only add the new children to the unsolved count, preserving solved count
            int newChildren = static_cast<int>(children.size() - oldSize);
            unsolvedChildCount.fetch_add(newChildren, std::memory_order_relaxed);
        }
    }
    
    /**
     * @brief Update child node type and check for solver propagation.
     * @param childIdx Index of the child
     * @param childType The child's proven node type
     * @return True if this node became solved as a result
     */
    bool update_child_node_type(int childIdx, NodeType childType) {
        if (!SearchParams::ENABLE_MCTS_SOLVER) return false;
        
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        
        if (nodeType != NodeType::UNSOLVED) {
            return false;  // Already solved
        }
        
        if (childIdx < 0 || static_cast<size_t>(childIdx) >= childNodeTypes.size()) {
            return false;
        }
        
        if (childNodeTypes[childIdx] != NodeType::UNSOLVED) {
            return false;  // Already recorded
        }
        
        childNodeTypes[childIdx] = childType;
        unsolvedChildCount.fetch_sub(1, std::memory_order_relaxed);
        
        // Check solver conditions (from child's perspective, so inverted)
        // If any child is a LOSS (for the child), this node is a WIN (we can force mate)
        if (childType == NodeType::LOSS) {
            nodeType = NodeType::WIN;
            // Use shortest path to win
            if (children[childIdx]) {
                endInPly = children[childIdx]->get_end_in_ply() + 1;
            }
            return true;
        }
        
        // If all expanded children are solved AND there are no more children to expand,
        // check if we're lost or drawn.
        // IMPORTANT: We can only mark as LOSS if ALL possible moves have been explored.
        // With progressive widening, there may be unexpanded moves that could save us.
        // CRITICAL: Must also verify the node is actually expanded (generator initialized).
        // An unexpanded node has an empty generator which would incorrectly pass hasNext() check.
        if (unsolvedChildCount.load(std::memory_order_relaxed) == 0 && 
            m_is_expanded && !candidateGenerator.hasNext()) {
            bool allWins = true;
            bool hasDrawn = false;
            int longestPly = 0;
            
            for (size_t i = 0; i < childNodeTypes.size(); i++) {
                if (childNodeTypes[i] != NodeType::WIN) {
                    allWins = false;
                }
                if (childNodeTypes[i] == NodeType::DRAW) {
                    hasDrawn = true;
                }
                if (children[i] && children[i]->get_end_in_ply() > longestPly) {
                    longestPly = children[i]->get_end_in_ply();
                }
            }
            
            if (allWins) {
                // All children are wins for them = loss for us
                nodeType = NodeType::LOSS;
                endInPly = longestPly + 1;  // Delay mate as long as possible
                return true;
            } else if (hasDrawn) {
                nodeType = NodeType::DRAW;
                return true;
            }
        }
        
        return false;
    }
    
    // =========================================================================
    // Q-Value Selection Methods
    // =========================================================================
    
    /**
     * @brief Get the Q-values for all expanded children.
     */
    std::vector<float> get_q_values() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return qValues;
    }
    
    /**
     * @brief Get a child's Q-value from the parent's perspective.
     * 
     * This returns the Q-value stored in this node's qValues array,
     * which is from the parent's (this node's) perspective, NOT the
     * child's own accumulated valueSum which would be from the opponent's
     * perspective.
     * 
     * @param childIdx Index of the child
     * @return Q-value from parent's perspective, or 0.0f if invalid index
     */
    float get_child_q(int childIdx) const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        if (childIdx >= 0 && static_cast<size_t>(childIdx) < qValues.size()) {
            return qValues[childIdx];
        }
        return 0.0f;
    }
    
    /**
     * @brief Get the index of the child with the highest Q-value.
     */
    int get_best_q_idx() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        if (qValues.empty()) return -1;
        
        int bestIdx = 0;
        float bestQ = qValues[0];
        for (size_t i = 1; i < qValues.size(); i++) {
            if (qValues[i] > bestQ) {
                bestQ = qValues[i];
                bestIdx = static_cast<int>(i);
            }
        }
        return bestIdx;
    }
    
    /**
     * @brief Get the best move index using Q-value veto and weighting.
     * 
     * Implements CrazyAra's Q-value veto: if the best Q-value move differs
     * significantly from the most-visited move, use Q-value to select.
     * 
     * @param qVetoDelta Threshold for Q-value veto (0 = disabled)
     * @param qValueWeight Weight for Q-value adjustment (0 = pure visits)
     * @return Index of the best move considering Q-values
     */
    int get_best_move_idx_with_q_weight(float qVetoDelta = SearchParams::Q_VETO_DELTA,
                                        float qValueWeight = SearchParams::Q_VALUE_WEIGHT) const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        
        if (childVisits.empty() || qValues.empty()) return -1;
        
        // Handle solved nodes
        if (nodeType == NodeType::WIN) {
            // Find the child that's a LOSS (quickest win)
            int bestIdx = -1;
            int shortestPly = INT32_MAX;
            for (size_t i = 0; i < childNodeTypes.size(); i++) {
                if (childNodeTypes[i] == NodeType::LOSS) {
                    if (children[i] && children[i]->get_end_in_ply() < shortestPly) {
                        shortestPly = children[i]->get_end_in_ply();
                        bestIdx = static_cast<int>(i);
                    }
                }
            }
            if (bestIdx >= 0) return bestIdx;
        }
        
        if (nodeType == NodeType::LOSS) {
            // Find the child with longest ply (delay mate)
            int bestIdx = 0;
            int longestPly = 0;
            for (size_t i = 0; i < children.size(); i++) {
                if (children[i] && children[i]->get_end_in_ply() > longestPly) {
                    longestPly = children[i]->get_end_in_ply();
                    bestIdx = static_cast<int>(i);
                }
            }
            return bestIdx;
        }
        
        // Find most-visited child
        int bestVisitIdx = 0;
        int maxVisits = childVisits[0];
        for (size_t i = 1; i < childVisits.size(); i++) {
            if (childVisits[i] > maxVisits) {
                maxVisits = childVisits[i];
                bestVisitIdx = static_cast<int>(i);
            }
        }
        
        // Find best Q-value child
        int bestQIdx = 0;
        float bestQ = qValues[0];
        for (size_t i = 1; i < qValues.size(); i++) {
            if (qValues[i] > bestQ) {
                bestQ = qValues[i];
                bestQIdx = static_cast<int>(i);
            }
        }
        
        // Q-value veto: if best-Q move is significantly better, use it
        if (qVetoDelta > 0.0f && bestQIdx != bestVisitIdx) {
            if (qValues[bestQIdx] > qValues[bestVisitIdx] + qVetoDelta && 
                childVisits[bestQIdx] > 1) {
                return bestQIdx;
            }
        }
        
        // Q-value weighting (for stochastic selection, not direct move choice)
        // For direct move selection, we just use visits or Q-veto
        return bestVisitIdx;
    }
    
    /**
     * @brief Get MCTS policy with Q-value adjustments.
     * 
     * Creates a policy distribution from visit counts with:
     * - Q-value veto: swaps visit counts when Q-value is much better
     * - Q-value weighting: transfers mass to higher-Q moves
     * - Prunes moves that lead to proven losses
     * 
     * @return Normalized policy vector
     */
    std::vector<float> get_mcts_policy_with_q_weight() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        
        if (childVisits.empty()) return {};
        
        std::vector<float> policy(childVisits.size());
        float total = 0.0f;
        
        // Start with visit counts
        for (size_t i = 0; i < childVisits.size(); i++) {
            policy[i] = static_cast<float>(childVisits[i]);
            total += policy[i];
        }
        
        if (total == 0.0f) return policy;
        
        // Prune moves leading to losses (if solver is enabled)
        if (SearchParams::ENABLE_MCTS_SOLVER && nodeType == NodeType::UNSOLVED) {
            for (size_t i = 0; i < childNodeTypes.size() && i < policy.size(); i++) {
                // Child WIN = bad for us (they win)
                if (childNodeTypes[i] == NodeType::WIN) {
                    total -= policy[i];
                    policy[i] = 0.0f;
                }
            }
        }
        
        // Find best-visited and second-best indices
        int bestVisitIdx = 0;
        int secondVisitIdx = -1;
        for (size_t i = 1; i < policy.size(); i++) {
            if (policy[i] > policy[bestVisitIdx]) {
                secondVisitIdx = bestVisitIdx;
                bestVisitIdx = static_cast<int>(i);
            } else if (secondVisitIdx < 0 || policy[i] > policy[secondVisitIdx]) {
                secondVisitIdx = static_cast<int>(i);
            }
        }
        
        // Find best Q-value index
        int bestQIdx = 0;
        if (!qValues.empty()) {
            for (size_t i = 1; i < qValues.size(); i++) {
                if (qValues[i] > qValues[bestQIdx]) {
                    bestQIdx = static_cast<int>(i);
                }
            }
        }
        
        // Q-value veto
        if (SearchParams::Q_VETO_DELTA > 0.0f && bestQIdx != bestVisitIdx) {
            if (qValues[bestQIdx] > qValues[bestVisitIdx] + SearchParams::Q_VETO_DELTA &&
                childVisits[bestQIdx] > 1) {
                // Swap visit counts
                std::swap(policy[bestQIdx], policy[bestVisitIdx]);
            }
        }
        // Q-value weighting (second-best boost)
        else if (SearchParams::Q_VALUE_WEIGHT > 0.0f && secondVisitIdx >= 0) {
            if (qValues[secondVisitIdx] > qValues[bestVisitIdx]) {
                float qDiff = qValues[secondVisitIdx] - qValues[bestVisitIdx];
                policy[secondVisitIdx] += qDiff * SearchParams::Q_VALUE_WEIGHT * policy[bestVisitIdx];
                total += qDiff * SearchParams::Q_VALUE_WEIGHT * policy[bestVisitIdx];
            }
        }
        
        // Normalize
        if (total > 0.0f) {
            for (auto& p : policy) {
                p /= total;
            }
        }
        
        return policy;
    }
};
