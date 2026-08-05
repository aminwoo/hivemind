#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <random>
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
    
    // Joint actions generated lazily in descending policy order
    JointCandidateGenerator candidateGenerator;
    int expandedCount = 0;
    
    // MCTS statistics
    std::vector<float> qValues;
    std::vector<float> childValueSum;
    std::vector<float> childPriors;
    std::vector<int> childVisits;
    std::vector<int> virtualLoss;  // Virtual visits for batch MCTS
    int virtualVisitSum = 0;
    float valueSum = 0.0f;
    std::atomic<int> m_depth{0};
    std::atomic<int> m_visits{0};  // Atomic for lock-free read/write
    std::atomic<bool> evaluationPending{false};
    bool m_is_expanded = false;

    Stockfish::Color teamToPlay;
    
    // MCGS (Monte Carlo Graph Search) support
    std::atomic<uint64_t> positionHash{0};  // Zobrist hash for transposition detection
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

    void update_and_remove_virtual_loss(size_t childIdx, float value) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        virtualLoss[childIdx]--;
        virtualVisitSum--;
        childVisits[childIdx]++;

        if (childVisits[childIdx] == 1) {
            childValueSum[childIdx] = value;
            qValues[childIdx] = value;
        } else {
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
        m_depth.store(value, std::memory_order_relaxed);
    }

    int get_depth() {
        return m_depth.load(std::memory_order_relaxed);
    }

    /**
     * @brief Returns the number of generated candidates (should equal children.size()).
     */
    size_t get_num_generated() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return candidateGenerator.generatedCount();
    }

    bool has_unexpanded_joint_actions() {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return candidateGenerator.hasNext();
    }

    bool should_expand_new_child(const SearchParams::RuntimeConfig& config) const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        for (size_t index = 0; index < childVisits.size(); index++) {
            const int virtualVisits = virtualLoss[index];
            if (childVisits[index] + virtualVisits == 0) {
                return false;
            }
        }
        const float coefficient = m_depth.load(std::memory_order_relaxed) == 0
            ? config.rootPwCoefficient
            : config.pwCoefficient;
        return candidateGenerator.hasNext()
            && expandedCount < SearchParams::get_allowed_children(
                m_visits.load(std::memory_order_relaxed) + virtualVisitSum,
                coefficient,
                config.pwExponent);
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
    * @brief Expands the next joint action candidate using fixed full-loss FPU.
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
                                                   JointActionCandidate& outAction,
                                                   const SearchParams::RuntimeConfig&,
                                                   int* outChildIdx = nullptr,
                                                   bool reserveForSelection = false) {
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
            // Node values are from the child's perspective; edge values are
            // stored from this parent's perspective.
            childQ = -existingNode->Q();
        } else {
            // Create new node
            child = std::make_shared<Node>(~teamToPlay, positionHash);
            child->set_depth(m_depth + 1);
            childQ = SearchParams::Q_INIT;
        }
        
        childValueSum.push_back(childQ);
        childPriors.push_back(candidate.jointPrior);
        // A transposition creates a new edge. Seed it with one pseudo-visit so
        // selection can use the shared node value without inheriting unrelated
        // visits accumulated through other parents.
        childVisits.push_back(existingNode ? 1 : 0);
        virtualLoss.push_back(reserveForSelection ? 1 : 0);
        if (reserveForSelection) {
            virtualVisitSum++;
        }
        children.push_back(child);
        qValues.push_back(childQ);
        
        expandedCount++;
        if (outChildIdx) {
            *outChildIdx = expandedCount - 1;
        }
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
                             bool teamHasTimeAdvantage,
                             bool boardAOnTurn,
                             bool boardBOnTurn,
                             const SearchParams::RuntimeConfig& config) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        
        // Already expanded by another thread
        if (m_is_expanded) {
            return false;
        }
        
        std::vector<float> rootPriorsA = priorsA;
        std::vector<float> rootPriorsB = priorsB;
        if (m_depth.load(std::memory_order_relaxed) == 0
            && config.rootDirichletAlpha > 0.0f
            && config.rootDirichletEpsilon > 0.0f) {
            auto applyNoise = [&](std::vector<float>& priors, uint64_t salt) {
                if (priors.size() <= 1) {
                    return;
                }
                std::mt19937_64 randomEngine(
                    config.rootNoiseSeed ^ positionHash.load(std::memory_order_relaxed) ^ salt);
                std::gamma_distribution<float> gamma(config.rootDirichletAlpha, 1.0f);
                std::vector<float> noise(priors.size());
                float total = 0.0f;
                for (float& sample : noise) {
                    sample = gamma(randomEngine);
                    total += sample;
                }
                if (total <= 0.0f) {
                    return;
                }
                const float epsilon = std::clamp(config.rootDirichletEpsilon, 0.0f, 1.0f);
                for (size_t index = 0; index < priors.size(); ++index) {
                    priors[index] = (1.0f - epsilon) * priors[index]
                        + epsilon * noise[index] / total;
                }
            };
            applyNoise(rootPriorsA, 0x9e3779b97f4a7c15ULL);
            applyNoise(rootPriorsB, 0xbf58476d1ce4e5b9ULL);
        }

        candidateGenerator.initialize(actionsA, actionsB, rootPriorsA, rootPriorsB,
                          teamHasTimeAdvantage, boardAOnTurn, boardBOnTurn);
        expandedCount = 0;
        
        // Try to expand the first child
        if (candidateGenerator.hasNext()) {
            JointActionCandidate candidate = candidateGenerator.getNext();

            auto child = std::make_shared<Node>(~teamToPlay);
            child->set_depth(m_depth + 1);

            childValueSum.push_back(SearchParams::Q_INIT);
            childPriors.push_back(candidate.jointPrior);
            childVisits.push_back(0);
            virtualLoss.push_back(0);
            children.push_back(child);
            qValues.push_back(SearchParams::Q_INIT);
            
            expandedCount++;
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
     * @brief Selects the best child and returns both the child pointer and index.
     * Thread-safe: Returns the child index atomically with the selection.
     * @return pair of (child pointer, child index), or (nullptr, -1) if no children
     */
    std::pair<std::shared_ptr<Node>, int> select_child_and_apply_virtual_loss(
        const SearchParams::RuntimeConfig& config = SearchParams::RuntimeConfig{});

    std::vector<std::shared_ptr<Node>> get_children() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return children;
    }

    void replace_child(int childIdx, const std::shared_ptr<Node>& child) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        if (childIdx >= 0 && static_cast<size_t>(childIdx) < children.size()) {
            children[childIdx] = child;
        }
    }

    bool is_expanded() {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return m_is_expanded;
    }

    bool try_reserve_evaluation() {
        bool expected = false;
        return evaluationPending.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel);
    }

    void release_evaluation_reservation() {
        evaluationPending.store(false, std::memory_order_release);
    }

    void set_value(float value) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        valueSum = value;
    }

    Stockfish::Color get_team_to_play() {
        return teamToPlay;
    }

    int get_visits() {
        return m_visits.load(std::memory_order_relaxed);
    }

    // Virtual loss methods for batch MCTS (thread-safe)
    void apply_virtual_loss(int childIdx, int amount = 1) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        if (childIdx >= 0 && static_cast<size_t>(childIdx) < virtualLoss.size()) {
            virtualLoss[childIdx] += amount;
            virtualVisitSum += amount;
        }
    }

    void remove_virtual_loss(int childIdx, int amount = 1) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        if (childIdx >= 0 && static_cast<size_t>(childIdx) < virtualLoss.size()) {
            virtualLoss[childIdx] -= amount;
            virtualVisitSum -= amount;
        }
    }

    float Q() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return valueSum / (1.0f + m_visits.load(std::memory_order_relaxed));
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
     * @brief Get the position hash for this node.
     * Used for transposition table lookups.
     */
    uint64_t get_hash() const {
        return positionHash.load(std::memory_order_relaxed);
    }
    
    /**
     * @brief Set the position hash for this node.
     */
    void set_hash(uint64_t hash) {
        positionHash.store(hash, std::memory_order_relaxed);
    }
    
    // =========================================================================
    // MCTS Solver Methods
    // =========================================================================
    
    /**
     * @brief Get the node type (UNSOLVED, WIN, LOSS, DRAW).
     */
    NodeType get_node_type() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return nodeType;
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
     * @brief Get the ply distance to terminal.
     */
    int get_end_in_ply() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return endInPly;
    }
    
    /**
     * @brief Initialize child node types array to match children count.
     * 
    * When new children are added, grow the childNodeTypes array and update
    * the unsolved count.
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
        // There may be unexpanded moves that could save us.
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
        int secondVisitIdx = -1;
        for (size_t i = 1; i < childVisits.size(); i++) {
            if (childVisits[i] > maxVisits) {
                secondVisitIdx = bestVisitIdx;
                maxVisits = childVisits[i];
                bestVisitIdx = static_cast<int>(i);
            } else if (secondVisitIdx < 0 || childVisits[i] > childVisits[secondVisitIdx]) {
                secondVisitIdx = static_cast<int>(i);
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

        if (qValueWeight > 0.0f && secondVisitIdx >= 0 &&
            qValues[secondVisitIdx] > qValues[bestVisitIdx]) {
            float qDifference = qValues[secondVisitIdx] - qValues[bestVisitIdx];
            float adjustedSecondVisits = childVisits[secondVisitIdx] +
                qDifference * qValueWeight * childVisits[bestVisitIdx];
            if (adjustedSecondVisits > childVisits[bestVisitIdx]) {
                return secondVisitIdx;
            }
        }
        
        // Q-value weighting (for stochastic selection, not direct move choice)
        // For direct move selection, we just use visits or Q-veto
        return bestVisitIdx;
    }
    
};
