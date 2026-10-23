#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <shared_mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include "environment/board.h"
#include "environment/constants.h"
#include "environment/joint_action.h"
#include "search/search_params.h"
#include "common/utils.h"
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

class Node : public std::enable_shared_from_this<Node> {
private:
    struct ParentEdge {
        std::weak_ptr<Node> parent;
        int childIdx;
    };

    // Reader-writer mutex for thread-safe access to node state
    // Allows multiple concurrent readers, exclusive writers
    mutable std::shared_mutex nodeMutex;
    
    // Children indexed by expansion order
    std::vector<std::shared_ptr<Node>> children;
    
    // Joint actions generated lazily in descending policy order
    JointCandidateGenerator candidateGenerator;
    int expandedCount = 0;
    // Sum of the expanded children's joint priors (prior-mass widening).
    float expandedPriorSum = 0.0f;
    
    // Keep the statistics updated together during selection and backup in one
    // compact record. The previous struct-of-arrays layout dirtied four
    // separate cache lines for one edge update and pulled unrelated edges into
    // exclusive ownership. Twenty-byte records keep the PUCT scan contiguous
    // while placing an edge's Q, value sum, visits, and virtual loss together.
    //
    // Selection reads these fields and backup writes them, so both used to take
    // this node's lock exclusively. Sixteen workers descending one shared tree
    // then serialized on the root: a four-GPU search measured the same node
    // rate as a single GPU. The hot fields are atomic so both paths run under a
    // shared lock instead, leaving only the rare unvisited->visited transition
    // - which moves an edge between two plain containers - exclusive.
    struct EdgeStats {
        std::atomic<float> q;
        std::atomic<float> valueSum;
        float prior;
        std::atomic<int> visits;
        std::atomic<int> virtualLoss;

        EdgeStats(float qInit, float valueSumInit, float priorInit,
                  int visitsInit, int virtualLossInit)
            : q(qInit), valueSum(valueSumInit), prior(priorInit),
              visits(visitsInit), virtualLoss(virtualLossInit) {}

        // Growing the vector copies elements, which only ever happens under the
        // exclusive lock, so an element-wise relaxed copy is enough.
        EdgeStats(const EdgeStats& other)
            : q(other.q.load(std::memory_order_relaxed)),
              valueSum(other.valueSum.load(std::memory_order_relaxed)),
              prior(other.prior),
              visits(other.visits.load(std::memory_order_relaxed)),
              virtualLoss(other.virtualLoss.load(std::memory_order_relaxed)) {}
        EdgeStats& operator=(const EdgeStats& other) {
            q.store(other.q.load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
            valueSum.store(other.valueSum.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
            prior = other.prior;
            visits.store(other.visits.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
            virtualLoss.store(other.virtualLoss.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
            return *this;
        }
    };
    static_assert(sizeof(EdgeStats) == 20);
    std::vector<EdgeStats> edges;

    template <typename Atomic>
    static float add_value(Atomic& sum, float value) {
        if constexpr (requires { sum.fetch_add(value, std::memory_order_relaxed); }) {
            return sum.fetch_add(value, std::memory_order_relaxed);
        } else {
            // Older Apple libc++ lacks floating-point fetch_add. Preserve the
            // same atomic update and relaxed ordering with compare/exchange.
            float previous = sum.load(std::memory_order_relaxed);
            while (!sum.compare_exchange_weak(previous, previous + value,
                                             std::memory_order_relaxed,
                                             std::memory_order_relaxed)) {}
            return previous;
        }
    }

    // Visits and the value sum each advance atomically and Q is republished
    // from them, so two concurrent backups on one edge can publish a Q built
    // from slightly mismatched (sum, visits) pairs - including a first visit's
    // store landing on top of a second visit's add. That is the usual
    // relaxation for lock-free MCTS backup: the error is bounded by one
    // evaluation and the next visit overwrites it.
    static void accumulate_edge_value(EdgeStats& edge, float value,
                                      bool exactFirstBackup = false) {
        const int visits =
            edge.visits.fetch_add(1, std::memory_order_relaxed) + 1;
        float sum;
        if (visits == 1 && exactFirstBackup) {
            // A new edge's sum holds the Q_INIT seed (a transposition's edge
            // starts at one visit and never takes this branch). Adding the
            // difference removes the seed without overwriting a concurrent
            // second backup's add.
            const float delta = value - SearchParams::Q_INIT;
            sum = add_value(edge.valueSum, delta) + delta;
        } else if (visits == 1) {
            // The first real visit replaces the FPU seed instead of adding to it.
            edge.valueSum.store(value, std::memory_order_relaxed);
            sum = value;
        } else {
            sum = add_value(edge.valueSum, value)
                + value;
        }
        // With exact backups, divide by the visits now counted rather than
        // this backup's own count: a concurrent backup's add may already be in
        // the sum. (Q is republished by every backup either way.)
        const int divisor = exactFirstBackup
            ? std::max(visits, edge.visits.load(std::memory_order_relaxed))
            : visits;
        edge.q.store(sum / static_cast<float>(divisor),
                     std::memory_order_relaxed);
    }

    std::atomic<int> virtualVisitSum{0};
    float visitedPolicySum = 0.0f;
    struct UnvisitedEdge {
        float prior;
        int childIdx;
    };
    struct UnvisitedEdgeOrder {
        bool operator()(const UnvisitedEdge& lhs,
                        const UnvisitedEdge& rhs) const {
            if (lhs.prior != rhs.prior) {
                return lhs.prior > rhs.prior;
            }
            return lhs.childIdx < rhs.childIdx;
        }
    };
    std::set<UnvisitedEdge, UnvisitedEdgeOrder> unvisitedEdges;
    std::vector<int> visitedEdges;
    std::vector<int> visitedEdgePositions;
    std::atomic<float> valueSum{0.0f};
    std::atomic<int> m_depth{0};
    std::atomic<int> m_visits{0};  // Atomic for lock-free read/write
    std::atomic<bool> evaluationPending{false};
    std::atomic<bool> m_is_expanded{false};

    Stockfish::Color teamToPlay;
    
    // MCGS (Monte Carlo Graph Search) support
    std::atomic<uint64_t> positionHash{0};  // Zobrist hash for transposition detection
    // MCTS Solver support
    std::atomic<NodeType> nodeType{NodeType::UNSOLVED};  // Proven game-theoretic value
    std::vector<NodeType> childNodeTypes;    // Cached node types of children
    std::atomic<int> unsolvedChildCount{0};  // Number of unsolved children (for solver)
    int provenWinningChildCount = 0;
    bool allowLossProof = true;
    std::atomic<int> endInPly{0};                        // Distance to terminal (for mate distance)

    struct RootGumbelEntry {
        int childIdx;
        int baselineVisits;
    };
    bool rootGumbelEnabled = false;
    float rootGumbelValueScale = SearchParams::ROOT_GUMBEL_VALUE_SCALE;
    int rootGumbelExpansionTarget = 0;
    int rootGumbelRoundQuota = 1;
    std::vector<RootGumbelEntry> rootGumbelActive;
    std::vector<int> rootGumbelWaiting;
    std::vector<float> childGumbelScores;

    // Reverse graph edges let a proof discovered through one transposition
    // update every parent of the canonical node, not just the active path.
    // Keep these on a separate mutex so proof propagation never nests two
    // nodes' state mutexes.
    mutable std::mutex parentEdgesMutex;
    std::vector<ParentEdge> parentEdges;

    void register_parent_edge(const std::shared_ptr<Node>& parent, int childIdx);
    void propagate_proof_to_parents();
    bool update_child_node_type_from(
        int childIdx, const Node* expectedChild, NodeType childType);
    void initialize_root_gumbel_locked(
        const SearchParams::RuntimeConfig& config);
    void replenish_root_gumbel_locked(
        const SearchParams::RuntimeConfig& config);
    void advance_root_gumbel_round_locked(
        const SearchParams::RuntimeConfig& config);
    void mark_edge_visited_locked(size_t childIdx) {
        unvisitedEdges.erase({edges[childIdx].prior,
                              static_cast<int>(childIdx)});
        visitedEdgePositions[childIdx] = static_cast<int>(visitedEdges.size());
        visitedEdges.push_back(static_cast<int>(childIdx));
        visitedPolicySum += edges[childIdx].prior;
    }
    // Root noise seeds, one per board, mixed with the search's noise seed and
    // the position hash.
    static constexpr uint64_t ROOT_NOISE_SALT_A = 0x9e3779b97f4a7c15ULL;
    static constexpr uint64_t ROOT_NOISE_SALT_B = 0xbf58476d1ce4e5b9ULL;
    // After edge priors are rewritten in place: rebuild what is keyed or summed
    // by them, so the unvisited set can still find each edge it erases.
    void reindex_edge_priors_locked() {
        unvisitedEdges.clear();
        visitedPolicySum = 0.0f;
        expandedPriorSum = 0.0f;
        for (size_t index = 0; index < edges.size(); ++index) {
            expandedPriorSum += edges[index].prior;
            if (visitedEdgePositions[index] >= 0) {
                visitedPolicySum += edges[index].prior;
            } else {
                unvisitedEdges.insert({edges[index].prior,
                                       static_cast<int>(index)});
            }
        }
    }
    void mark_edge_unvisited_locked(size_t childIdx) {
        visitedPolicySum -= edges[childIdx].prior;
        const int position = visitedEdgePositions[childIdx];
        const int lastChildIdx = visitedEdges.back();
        visitedEdges[position] = lastChildIdx;
        visitedEdgePositions[static_cast<size_t>(lastChildIdx)] = position;
        visitedEdges.pop_back();
        visitedEdgePositions[childIdx] = -1;
        unvisitedEdges.insert({edges[childIdx].prior,
                               static_cast<int>(childIdx)});
    }

    // First-play Q for an unvisited edge. Caller holds nodeMutex.
    float selection_fpu_q_locked(const SearchParams::RuntimeConfig& config,
                                 int visits) const {
        if (!config.enableDynamicFpu || visits <= 0) {
            return SearchParams::Q_INIT;
        }
        const float parentQ = valueSum / static_cast<float>(visits);
        return std::clamp(
            parentQ - config.fpuReduction
                * std::sqrt(std::max(0.0f, visitedPolicySum)),
            parentQ - 1.0f, parentQ + 1.0f);
    }

    // An edge's Q as PUCT selection sees it, virtual loss included. Takes the
    // caller's snapshot of visits and virtual loss so Q and U agree.
    static float selection_q(const EdgeStats& edge, uint32_t n, int vl, float fpuQ) {
        const uint32_t nEffective = n + static_cast<uint32_t>(vl);
        if (nEffective == 0) {
            return fpuQ;
        }
        if (vl == 0) {
            return edge.q.load(std::memory_order_relaxed);
        }
        switch (SearchParams::get_virtual_style(n)) {
        case SearchParams::VirtualStyle::VIRTUAL_LOSS:
            return (edge.valueSum.load(std::memory_order_relaxed)
                    - static_cast<float>(vl))
                / static_cast<float>(nEffective);
        case SearchParams::VirtualStyle::VIRTUAL_OFFSET:
            return edge.q.load(std::memory_order_relaxed)
                - static_cast<float>(vl)
                * static_cast<float>(SearchParams::VIRTUAL_OFFSET_STRENGTH);
        default:
            return edge.q.load(std::memory_order_relaxed);
        }
    }

    // PUCT widening: does the next ungenerated joint action outscore every
    // visited child, scored as selection would score them (mate-hint bonus
    // included)? Caller holds nodeMutex (shared is enough).
    //
    // Approximate once the FPU is factored: only the next action in prior
    // order is tested, and a later one with a better estimate waits for it.
    bool next_action_wins_puct_locked(const SearchParams::RuntimeConfig& config,
                                      int visits,
                                      const JointActionCandidate* preferredAction) const {
        if (visits <= 0 || visitedEdges.empty()) {
            return false;
        }
        const JointActionCandidate next = candidateGenerator.peekNext();
        if (next.jointPrior <= 0.0f) {
            return false;
        }
        const float explorationBase = SearchParams::get_cpuct(
            static_cast<float>(visits), config.cpuctInit, config.cpuctBase)
            * std::sqrt(static_cast<float>(visits));
        const float fpuQ = selection_fpu_q_locked(config, visits);
        const bool hasNonLosingAlternative =
            nodeType.load(std::memory_order_relaxed) == NodeType::UNSOLVED
            && provenWinningChildCount < static_cast<int>(children.size());
        const bool factored = config.factoredFpuVisits > 0.0f;
        float bestScore = -std::numeric_limits<float>::infinity();
        // Visit-weighted value sums: all visited edges, and those sharing the
        // next action's board-A or board-B move.
        float sumAll = 0.0f, sumA = 0.0f, sumB = 0.0f;
        float visitsAll = 0.0f, visitsA = 0.0f, visitsB = 0.0f;
        for (int childIdx : visitedEdges) {
            if (childIdx < 0 || static_cast<size_t>(childIdx) >= children.size()
                || !children[childIdx]) {
                continue;
            }
            const EdgeStats& edge = edges[childIdx];
            const int vl = edge.virtualLoss.load(std::memory_order_relaxed);
            const uint32_t n = static_cast<uint32_t>(
                edge.visits.load(std::memory_order_relaxed));
            const JointActionCandidate& action =
                candidateGenerator.getGenerated(static_cast<size_t>(childIdx));
            // A proven-losing child is not selectable but is still evidence.
            if (!(hasNonLosingAlternative
                  && children[childIdx]->get_node_type() == NodeType::WIN)) {
                const float nEffective = static_cast<float>(n + static_cast<uint32_t>(vl));
                float score = selection_q(edge, n, vl, fpuQ)
                    + explorationBase * edge.prior / (1.0f + nEffective);
                if (preferredAction && action.moveA == preferredAction->moveA
                    && action.moveB == preferredAction->moveB) {
                    score += config.internalMateProbeBias
                        / (1.0f + nEffective
                            / std::max(config.internalMateProbeBiasDecayVisits, 1.0f));
                }
                bestScore = std::max(bestScore, score);
            }
            if (factored && n > 0) {
                const float weight = static_cast<float>(n);
                const float weightedQ =
                    weight * edge.q.load(std::memory_order_relaxed);
                sumAll += weightedQ;
                visitsAll += weight;
                if (action.moveA == next.moveA) {
                    sumA += weightedQ;
                    visitsA += weight;
                }
                if (action.moveB == next.moveB) {
                    sumB += weightedQ;
                    visitsB += weight;
                }
            }
        }
        float nextQ = fpuQ;
        if (factored && (visitsA > 0.0f || visitsB > 0.0f)) {
            const bool both = visitsA > 0.0f && visitsB > 0.0f;
            const float estimate = both
                ? sumA / visitsA + sumB / visitsB - sumAll / visitsAll
                : visitsA > 0.0f ? sumA / visitsA : sumB / visitsB;
            // The additive estimate is only as sure as its weaker marginal.
            const float support = both
                ? 1.0f / (1.0f / visitsA + 1.0f / visitsB)
                : visitsA + visitsB;
            nextQ += support / (support + config.factoredFpuVisits)
                * (std::clamp(estimate, -1.0f, 1.0f) - fpuQ);
        }
        return nextQ + explorationBase * next.jointPrior > bestScore;
    }

public:
    struct ChildSelection {
        std::shared_ptr<Node> child;
        int childIdx = -1;
        bool hasEvaluationReservation = false;
        std::shared_ptr<Node> pendingEvaluation;
    };

    static inline std::atomic<long> s_liveNodes{0};
    static long live_count() { return s_liveNodes.load(std::memory_order_relaxed); }

    Node(Stockfish::Color teamToPlay) : teamToPlay(teamToPlay) { s_liveNodes.fetch_add(1, std::memory_order_relaxed); }
    Node(Stockfish::Color teamToPlay, uint64_t hash) 
        : teamToPlay(teamToPlay), positionHash(hash) { s_liveNodes.fetch_add(1, std::memory_order_relaxed); }
    ~Node() { s_liveNodes.fetch_sub(1, std::memory_order_relaxed); }

    // Lock the node for exclusive access
    std::unique_lock<std::shared_mutex> lock() {
        return std::unique_lock<std::shared_mutex>(nodeMutex);
    }

    void update(size_t childIdx, float value, bool exactFirstBackup = false) {
        {
            std::shared_lock<std::shared_mutex> guard(nodeMutex);
            EdgeStats& edge = edges[childIdx];
            if (edge.visits.load(std::memory_order_relaxed)
                    + edge.virtualLoss.load(std::memory_order_relaxed) != 0) {
                accumulate_edge_value(edge, value, exactFirstBackup);
                add_value(valueSum, value);
                m_visits.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        // First touch of this edge moves it between the visited and unvisited
        // containers, so take the lock exclusively and re-check under it.
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        EdgeStats& edge = edges[childIdx];
        if (visitedEdgePositions[childIdx] < 0
            && edge.visits.load(std::memory_order_relaxed)
                + edge.virtualLoss.load(std::memory_order_relaxed) == 0) {
            mark_edge_visited_locked(childIdx);
        }
        accumulate_edge_value(edge, value, exactFirstBackup);
        add_value(valueSum, value);
        m_visits.fetch_add(1, std::memory_order_relaxed);
    }

    // The edge already carries this thread's virtual loss, so it is already in
    // the visited set and no container changes here.
    void update_and_remove_virtual_loss(size_t childIdx, float value,
                                        bool exactFirstBackup = false) {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        EdgeStats& edge = edges[childIdx];
        edge.virtualLoss.fetch_sub(1, std::memory_order_relaxed);
        virtualVisitSum.fetch_sub(1, std::memory_order_relaxed);
        accumulate_edge_value(edge, value, exactFirstBackup);
        add_value(valueSum, value);
        m_visits.fetch_add(1, std::memory_order_relaxed);
    }

    void update_terminal(float value) {
        add_value(valueSum, value);
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

    /**
     * @brief The network's prior for @p move on @p boardNum at this node, or
     *        zero before the node is expanded or for a move it has no prior
     *        for. The priors are fixed once the node is expanded.
     */
    float move_prior(int boardNum, Stockfish::Move move) const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        if (!m_is_expanded.load(std::memory_order_acquire)) {
            return 0.0f;
        }
        return candidateGenerator.boardPrior(boardNum, move);
    }

    /**
     * @brief The child reached by the joint action (@p moveA, @p moveB), or
     *        null when this node has not generated that action or has not
     *        expanded its child.
     */
    std::shared_ptr<Node> child_for_action(Stockfish::Move moveA,
                                           Stockfish::Move moveB) const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        if (!m_is_expanded.load(std::memory_order_acquire)) {
            return nullptr;
        }
        const size_t generated = candidateGenerator.generatedCount();
        for (size_t index = 0; index < generated && index < children.size();
             ++index) {
            const JointActionCandidate& action =
                candidateGenerator.getGenerated(index);
            if (action.moveA == moveA && action.moveB == moveB) {
                return children[index];
            }
        }
        return nullptr;
    }

    /**
     * @brief Returns how many edges a PUCT selection scans at this node.
     *
     * Selection walks every visited edge plus at most one unvisited edge, all
     * under this node's exclusive lock, so at the root this is the length of
     * the critical section every worker serializes on.
     */
    size_t get_visited_edge_count() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return visitedEdges.size();
    }

    bool has_unexpanded_joint_actions() {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return candidateGenerator.hasNext();
    }

    bool should_expand_new_child(const SearchParams::RuntimeConfig& config,
                                 const JointActionCandidate* preferredAction = nullptr) const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return should_expand_new_child_locked(config, preferredAction);
    }

    bool should_expand_new_child_locked(const SearchParams::RuntimeConfig& config,
                                        const JointActionCandidate* preferredAction) const {
        if (candidateGenerator.hasPromoted()) {
            return true;
        }
        const bool allExpandedChildrenLose = !children.empty()
            && provenWinningChildCount == static_cast<int>(children.size());
        if (candidateGenerator.hasNext() && allExpandedChildrenLose) {
            return true;
        }
        if (rootGumbelEnabled) {
            return candidateGenerator.pendingGumbelCount() > 0
                && expandedCount < rootGumbelExpansionTarget;
        }
        if (!unvisitedEdges.empty()) {
            return false;
        }
        const float coefficient = m_depth.load(std::memory_order_relaxed) == 0
            ? config.rootPwCoefficient
            : config.pwCoefficient;
        const int visits = m_visits.load(std::memory_order_relaxed) + virtualVisitSum;
        const int countLimit = SearchParams::get_allowed_children(
            visits, coefficient, config.pwExponent);
        if (!candidateGenerator.hasNext()) {
            return false;
        }
        const float massStart = m_depth.load(std::memory_order_relaxed) == 0
            && config.rootPwMassStart >= 0.0f ? config.rootPwMassStart : config.pwMassStart;
        const float legalMass = config.pwMassNormalize ? candidateGenerator.legal_prior_mass() : 1.0f;
        bool widen;
        if (massStart <= 0.0f || legalMass <= 0.0f) {
            widen = expandedCount < countLimit;
        } else {
            const int cap = static_cast<int>(std::ceil(config.pwMassCap * static_cast<float>(countLimit)));
            widen = expandedCount < std::max(1, cap)
                && expandedPriorSum < legalMass * SearchParams::get_widening_mass_target(
                    visits, massStart, config.pwMassExponent);
        }
        return widen || (config.puctWidening
                         && next_action_wins_puct_locked(config, visits, preferredAction));
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

    bool promote_joint_action(const JointActionCandidate& action) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        return candidateGenerator.promote(action.moveA, action.moveB);
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
                                                   const SearchParams::RuntimeConfig& config,
                                                   int* outChildIdx = nullptr,
                                                   bool reserveForSelection = false,
                                                   bool* outEvaluationReserved = nullptr,
                                                   bool recheckWidening = false,
                                                   const JointActionCandidate* preferredAction = nullptr) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        if (outEvaluationReserved) {
            *outEvaluationReserved = false;
        }
        if (!candidateGenerator.hasNext()) {
            return nullptr;
        }
        // Score-based widening is decided under a shared lock, so another
        // worker may have opened the approved action since; recheck before
        // opening the next one. (Count widening keeps its old behaviour.)
        if (config.puctWidening && recheckWidening
            && !should_expand_new_child_locked(config, preferredAction)) {
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

        if (reserveForSelection) {
            const bool reserved = child->try_reserve_evaluation();
            if (!reserved) {
                return nullptr;
            }
            if (outEvaluationReserved) {
                *outEvaluationReserved = true;
            }
        }
        
        // A transposition creates a new edge. Seed it with one pseudo-visit so
        // selection can use the shared node value without inheriting unrelated
        // visits accumulated through other parents.
        edges.push_back({childQ, childQ, candidate.jointPrior,
                         existingNode ? 1 : 0,
                         reserveForSelection ? 1 : 0});
        visitedEdgePositions.push_back(-1);
        const int childIdx = expandedCount;
        if (existingNode || reserveForSelection) {
            visitedPolicySum += edges.back().prior;
            visitedEdgePositions.back() = static_cast<int>(visitedEdges.size());
            visitedEdges.push_back(childIdx);
        } else {
            unvisitedEdges.insert({edges.back().prior, childIdx});
        }
        if (reserveForSelection) {
            virtualVisitSum++;
        }
        children.push_back(child);
        childGumbelScores.push_back(candidate.gumbelScore);
        if (rootGumbelEnabled) {
            rootGumbelActive.push_back({
                expandedCount,
                edges.back().visits});
        }
        
        expandedCount++;
        expandedPriorSum += candidate.jointPrior;
        if (outChildIdx) {
            *outChildIdx = expandedCount - 1;
        }
        // That candidate may have been the last one. If so, the generator's
        // frontier machinery is spent and can go back now rather than sitting
        // in the tree until the search ends; it is rebuilt from the retained
        // move lists if this node is ever re-rooted.
        candidateGenerator.releaseExhaustedFrontier();
        const int registeredChildIdx = expandedCount - 1;
        std::shared_ptr<Node> parent = weak_from_this().lock();
        guard.unlock();
        child->register_parent_edge(parent, registeredChildIdx);
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
                             const SearchParams::RuntimeConfig& config,
                             const std::vector<uint8_t>& capturesA = {},
                             const std::vector<uint8_t>& capturesB = {},
                             const std::vector<float>& jointFactorsA = {},
                             const std::vector<float>& jointFactorsB = {},
                             size_t jointFactorRank = 0) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        
        // Already expanded by another thread
        if (m_is_expanded.load(std::memory_order_relaxed)) {
            return false;
        }
        
        std::vector<float> rootPriorsA = priorsA;
        std::vector<float> rootPriorsB = priorsB;
        if (m_depth.load(std::memory_order_relaxed) == 0
            && !config.enableGumbelRootSearch
            && config.rootDirichletAlpha > 0.0f
            && config.rootDirichletEpsilon > 0.0f) {
            const uint64_t hash = positionHash.load(std::memory_order_relaxed);
            mix_dirichlet_noise(rootPriorsA, config.rootDirichletAlpha,
                                config.rootDirichletEpsilon,
                                config.rootNoiseSeed ^ hash ^ ROOT_NOISE_SALT_A);
            mix_dirichlet_noise(rootPriorsB, config.rootDirichletAlpha,
                                config.rootDirichletEpsilon,
                                config.rootNoiseSeed ^ hash ^ ROOT_NOISE_SALT_B);
        }

        candidateGenerator.initialize(actionsA, actionsB, rootPriorsA, rootPriorsB,
                          teamHasTimeAdvantage, boardAOnTurn, boardBOnTurn,
                          capturesA, capturesB, jointFactorsA, jointFactorsB,
                          jointFactorRank,
                          static_cast<size_t>(std::max(
                              0, m_depth.load(std::memory_order_relaxed) == 0
                                  ? config.rootJointPolicyTopK
                                  : config.jointPolicyTopK)),
                          config.jointPolicyResidualScale);
        expandedCount = 0;
        expandedPriorSum = 0.0f;
        visitedPolicySum = 0.0f;
        unvisitedEdges.clear();
        visitedEdges.clear();
        visitedEdgePositions.clear();
        provenWinningChildCount = 0;
        initialize_root_gumbel_locked(config);
        
        // Try to expand the first child
        if (candidateGenerator.hasNext()) {
            JointActionCandidate candidate = candidateGenerator.getNext();

            auto child = std::make_shared<Node>(~teamToPlay);
            child->set_depth(m_depth + 1);

            edges.push_back({SearchParams::Q_INIT, SearchParams::Q_INIT,
                             candidate.jointPrior, 0, 0});
            visitedEdgePositions.push_back(-1);
            unvisitedEdges.insert({candidate.jointPrior, 0});
            children.push_back(child);
            childGumbelScores.push_back(candidate.gumbelScore);
            if (rootGumbelEnabled) {
                rootGumbelActive.push_back({0, 0});
            }
            
            expandedCount++;
            expandedPriorSum += candidate.jointPrior;
            m_is_expanded.store(true, std::memory_order_release);
            std::shared_ptr<Node> parent = weak_from_this().lock();
            guard.unlock();
            child->register_parent_edge(parent, 0);
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
    ChildSelection select_child_and_apply_virtual_loss(
        const SearchParams::RuntimeConfig& config = SearchParams::RuntimeConfig{},
        const std::unordered_set<const Node*>* blockedNodes = nullptr,
        const JointActionCandidate* preferredAction = nullptr);

    std::vector<std::shared_ptr<Node>> get_children() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        return children;
    }

    std::shared_ptr<Node> get_child(int childIdx) const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        if (childIdx < 0 || static_cast<size_t>(childIdx) >= children.size()) {
            return nullptr;
        }
        return children[childIdx];
    }

    void append_child_ptrs(std::vector<Node*>& output) const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        output.reserve(output.size() + children.size());
        for (const std::shared_ptr<Node>& child : children) {
            if (child) {
                output.push_back(child.get());
            }
        }
    }

    void replace_child(int childIdx, const std::shared_ptr<Node>& child) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        if (childIdx >= 0 && static_cast<size_t>(childIdx) < children.size()) {
            children[childIdx] = child;
            if (nodeType.load(std::memory_order_relaxed) == NodeType::UNSOLVED
                && static_cast<size_t>(childIdx) < childNodeTypes.size()
                && childNodeTypes[childIdx] != NodeType::UNSOLVED) {
                if (childNodeTypes[childIdx] == NodeType::WIN) {
                    provenWinningChildCount--;
                }
                childNodeTypes[childIdx] = NodeType::UNSOLVED;
                unsolvedChildCount.fetch_add(1, std::memory_order_relaxed);
            }
            std::shared_ptr<Node> parent = weak_from_this().lock();
            guard.unlock();
            child->register_parent_edge(parent, childIdx);
        }
    }

    bool is_expanded() const {
        return m_is_expanded.load(std::memory_order_acquire);
    }

    bool try_reserve_evaluation() {
        bool expected = false;
        return evaluationPending.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel);
    }

    bool is_evaluation_pending() const {
        return evaluationPending.load(std::memory_order_acquire);
    }

    void release_evaluation_reservation() {
        evaluationPending.store(false, std::memory_order_release);
        evaluationPending.notify_all();
    }

    void wait_for_evaluation_completion() const {
        evaluationPending.wait(true, std::memory_order_acquire);
    }

    /**
     * @brief Waits for a pending evaluation, giving up when the caller expires.
     *
     * `isExpired` is polled while waiting and returning true abandons the wait
     * (reported as false). std::atomic::wait has no timed form, so the bounded
     * variant polls; the interval is orders of magnitude below one inference,
     * which is what the wait is covering.
     */
    template <typename ExpiredFn>
    bool wait_for_evaluation_completion_until(ExpiredFn isExpired) const {
        constexpr auto POLL_INTERVAL = std::chrono::microseconds(100);
        while (evaluationPending.load(std::memory_order_acquire)) {
            if (isExpired()) {
                return false;
            }
            std::this_thread::sleep_for(POLL_INTERVAL);
        }
        return true;
    }

    void set_value(float value) {
        valueSum.store(value, std::memory_order_relaxed);
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
        if (childIdx >= 0 && static_cast<size_t>(childIdx) < edges.size()) {
            EdgeStats& edge = edges[childIdx];
            if (amount > 0 && edge.visits + edge.virtualLoss == 0) {
                mark_edge_visited_locked(static_cast<size_t>(childIdx));
            }
            edge.virtualLoss += amount;
            virtualVisitSum += amount;
        }
    }

    void remove_virtual_loss(int childIdx, int amount = 1) {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        if (childIdx >= 0 && static_cast<size_t>(childIdx) < edges.size()) {
            EdgeStats& edge = edges[childIdx];
            edge.virtualLoss -= amount;
            virtualVisitSum -= amount;
            if (edge.visits + edge.virtualLoss == 0) {
                mark_edge_unvisited_locked(static_cast<size_t>(childIdx));
            }
        }
    }

    float Q() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        const NodeType type = nodeType.load(std::memory_order_relaxed);
        if (type == NodeType::WIN) {
            return 1.0f;
        }
        if (type == NodeType::LOSS) {
            return -1.0f;
        }
        if (type == NodeType::DRAW) {
            return 0.0f;
        }
        const int visits = m_visits.load(std::memory_order_relaxed);
        const float sum = valueSum.load(std::memory_order_relaxed);
        return visits > 0 ? sum / static_cast<float>(visits) : sum;
    }
    
    /**
     * @brief Get the visit counts for all expanded children.
     * Used for extracting MCTS policy distributions.
     */
    std::vector<int> get_child_visits() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        std::vector<int> visits;
        visits.reserve(edges.size());
        for (const EdgeStats& edge : edges) {
            visits.push_back(edge.visits);
        }
        return visits;
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

    /**
     * @brief Configure a newly created or reused node as this search's root.
     * @param allowProvenLoss False only when another live board must keep
     * searching after its partner board has already been mated.
     */
    void configure_root_search(
        const SearchParams::RuntimeConfig& config,
        bool allowProvenLoss = true);

    /**
     * @brief Give a retained node the Dirichlet noise a fresh root gets.
     *
     * A node expanded below the root carries clean priors. Call this before
     * configure_root_search when such a node becomes the root of a search with
     * root noise: that call then rescores the expanded edges and rebuilds the
     * frontier from the noisy priors.
     */
    void refresh_root_noise(const SearchParams::RuntimeConfig& config);

    // =========================================================================
    // MCTS Solver Methods
    // =========================================================================
    
    /**
     * @brief Get the node type (UNSOLVED, WIN, LOSS, DRAW).
     */
    NodeType get_node_type() const {
        return nodeType.load(std::memory_order_acquire);
    }
    
    /**
     * @brief Mark this node as a WIN (opponent is mated).
     */
    void mark_as_win(int ply = 0);
    
    /**
     * @brief Mark this node as a LOSS (we are mated).
     */
    void mark_as_loss(int ply = 0);

    /**
     * @brief Mark this node as a proven draw.
     */
    void mark_as_draw(int ply = 0);
    
    /**
     * @brief Get the ply distance to terminal.
     */
    int get_end_in_ply() const {
        return endInPly.load(std::memory_order_relaxed);
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
        return update_child_node_type_from(childIdx, nullptr, childType);
    }
    
    // =========================================================================
    // Q-Value Selection Methods
    // =========================================================================
    
    /**
     * @brief Get the Q-values for all expanded children.
     */
    std::vector<float> get_q_values() const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        std::vector<float> values;
        values.reserve(edges.size());
        for (const EdgeStats& edge : edges) {
            values.push_back(edge.q);
        }
        return values;
    }
    
    /**
     * @brief Get a child's Q-value from the parent's perspective.
     * 
     * This returns the Q-value stored in this node's edge record,
     * which is from the parent's (this node's) perspective, NOT the
     * child's own accumulated valueSum which would be from the opponent's
     * perspective.
     * 
     * @param childIdx Index of the child
     * @return Q-value from parent's perspective, or 0.0f if invalid index
     */
    float get_child_q(int childIdx) const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        if (childIdx >= 0 && static_cast<size_t>(childIdx) < edges.size()) {
            return edges[childIdx].q;
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
    int get_best_move_idx_with_q_weight(
        float qVetoDelta = SearchParams::Q_VETO_DELTA,
        float qValueWeight = SearchParams::Q_VALUE_WEIGHT,
        bool avoidSolvedDraw = false) const {
        std::shared_lock<std::shared_mutex> guard(nodeMutex);
        
        if (edges.empty()) return -1;
        
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
        
        // A proven loss does not make mate distance a useful move-quality
        // signal for a live bughouse seat. Keep using the searched visits and
        // Q values below, so a dead partner board cannot make this board throw
        // away its strongest move merely to delay the combined mate score.

        const bool hasNonDrawingAlternative = avoidSolvedDraw && std::any_of(
            children.begin(), children.end(),
            [](const std::shared_ptr<Node>& child) {
                return child
                    && child->get_node_type() != NodeType::WIN
                    && child->get_node_type() != NodeType::DRAW;
            });

        if (rootGumbelEnabled && !rootGumbelActive.empty()) {
            int bestIdx = -1;
            float bestScore = -std::numeric_limits<float>::infinity();
            for (const RootGumbelEntry& entry : rootGumbelActive) {
                const int childIdx = entry.childIdx;
                if (childIdx < 0
                    || static_cast<size_t>(childIdx) >= children.size()
                    || !children[childIdx]
                    || children[childIdx]->get_node_type() == NodeType::WIN
                    || (hasNonDrawingAlternative
                        && children[childIdx]->get_node_type()
                            == NodeType::DRAW)) {
                    continue;
                }
                const NodeType childType = children[childIdx]->get_node_type();
                const float solverQ = childType == NodeType::LOSS
                    ? 1.0f
                    : childType == NodeType::DRAW
                        ? 0.0f
                        : edges[childIdx].q.load(std::memory_order_relaxed);
                const float score = childGumbelScores[childIdx]
                    + rootGumbelValueScale * solverQ;
                if (score > bestScore) {
                    bestScore = score;
                    bestIdx = childIdx;
                }
            }
            if (bestIdx >= 0) {
                return bestIdx;
            }
        }

        const bool hasNonLosingAlternative = std::any_of(
            children.begin(), children.end(),
            [](const std::shared_ptr<Node>& child) {
                return child && child->get_node_type() != NodeType::WIN;
            });
        auto isEligible = [&](size_t index) {
            if (!children[index]) {
                return true;
            }
            const NodeType childType = children[index]->get_node_type();
            if (hasNonLosingAlternative && childType == NodeType::WIN) {
                return false;
            }
            return !hasNonDrawingAlternative || childType != NodeType::DRAW;
        };

        size_t firstEligibleIdx = 0;
        while (firstEligibleIdx < edges.size() && !isEligible(firstEligibleIdx)) {
            ++firstEligibleIdx;
        }
        if (firstEligibleIdx == edges.size()) return -1;
        
        // Find most-visited child
        int bestVisitIdx = static_cast<int>(firstEligibleIdx);
        int maxVisits = edges[firstEligibleIdx].visits;
        int secondVisitIdx = -1;
        for (size_t i = firstEligibleIdx + 1; i < edges.size(); i++) {
            if (!isEligible(i)) continue;
            if (edges[i].visits > maxVisits) {
                secondVisitIdx = bestVisitIdx;
                maxVisits = edges[i].visits;
                bestVisitIdx = static_cast<int>(i);
            } else if (secondVisitIdx < 0
                       || edges[i].visits > edges[secondVisitIdx].visits) {
                secondVisitIdx = static_cast<int>(i);
            }
        }
        
        // Find best Q-value child
        int bestQIdx = static_cast<int>(firstEligibleIdx);
        float bestQ = edges[firstEligibleIdx].q;
        for (size_t i = firstEligibleIdx + 1; i < edges.size(); i++) {
            if (!isEligible(i)) continue;
            if (edges[i].q > bestQ) {
                bestQ = edges[i].q;
                bestQIdx = static_cast<int>(i);
            }
        }
        
        // Q-value veto: if best-Q move is significantly better, use it
        if (qVetoDelta > 0.0f && bestQIdx != bestVisitIdx) {
            if (edges[bestQIdx].q > edges[bestVisitIdx].q + qVetoDelta
                && edges[bestQIdx].visits > 1) {
                return bestQIdx;
            }
        }

        if (qValueWeight > 0.0f && secondVisitIdx >= 0 &&
            edges[secondVisitIdx].q > edges[bestVisitIdx].q) {
            float qDifference = edges[secondVisitIdx].q - edges[bestVisitIdx].q;
            float adjustedSecondVisits = edges[secondVisitIdx].visits
                + qDifference * qValueWeight * edges[bestVisitIdx].visits;
            if (adjustedSecondVisits > edges[bestVisitIdx].visits) {
                return secondVisitIdx;
            }
        }
        
        // Q-value weighting (for stochastic selection, not direct move choice)
        // For direct move selection, we just use visits or Q-veto
        return bestVisitIdx;
    }
    
};
