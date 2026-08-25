#include "search/node.h"
#include "search/search_params.h"
#include <limits>
#include <algorithm>

void Node::register_parent_edge(
    const std::shared_ptr<Node>& parent, int childIdx) {
    if (!SearchParams::ENABLE_MCTS_SOLVER || !parent || parent.get() == this
        || childIdx < 0) {
        return;
    }

    {
        std::lock_guard<std::mutex> guard(parentEdgesMutex);
        bool alreadyRegistered = false;
        auto edge = parentEdges.begin();
        while (edge != parentEdges.end()) {
            std::shared_ptr<Node> existingParent = edge->parent.lock();
            if (!existingParent) {
                edge = parentEdges.erase(edge);
                continue;
            }
            if (existingParent.get() == parent.get()
                && edge->childIdx == childIdx) {
                alreadyRegistered = true;
            }
            ++edge;
        }
        if (!alreadyRegistered) {
            parentEdges.push_back({parent, childIdx});
        }
    }

    // Register first, then sample the proof state. This handshake cannot miss a
    // concurrent solve: either this read observes it, or the solving thread
    // observes the newly registered reverse edge.
    const NodeType solvedType = get_node_type();
    if (solvedType != NodeType::UNSOLVED) {
        parent->update_child_node_type_from(childIdx, this, solvedType);
    }
}

void Node::propagate_proof_to_parents() {
    if (!SearchParams::ENABLE_MCTS_SOLVER) {
        return;
    }

    const NodeType solvedType = get_node_type();
    if (solvedType == NodeType::UNSOLVED) {
        return;
    }

    std::vector<ParentEdge> edges;
    {
        std::lock_guard<std::mutex> guard(parentEdgesMutex);
        auto edge = parentEdges.begin();
        while (edge != parentEdges.end()) {
            if (edge->parent.expired()) {
                edge = parentEdges.erase(edge);
            } else {
                ++edge;
            }
        }
        edges = parentEdges;
    }

    // Never hold this node's mutex while updating a parent. Each parent accepts
    // a child proof at most once, which also terminates duplicate paths/cycles.
    for (const ParentEdge& edge : edges) {
        if (std::shared_ptr<Node> parent = edge.parent.lock()) {
            parent->update_child_node_type_from(
                edge.childIdx, this, solvedType);
        }
    }
}

void Node::mark_as_win(int ply) {
    {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        if (nodeType.load(std::memory_order_relaxed) != NodeType::UNSOLVED) {
            return;
        }
        valueSum = static_cast<float>(m_visits.load(std::memory_order_relaxed) + 1);
        endInPly.store(ply, std::memory_order_relaxed);
        nodeType.store(NodeType::WIN, std::memory_order_release);
    }
    propagate_proof_to_parents();
}

void Node::mark_as_loss(int ply) {
    {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        if (nodeType.load(std::memory_order_relaxed) != NodeType::UNSOLVED) {
            return;
        }
        valueSum = -static_cast<float>(m_visits.load(std::memory_order_relaxed) + 1);
        endInPly.store(ply, std::memory_order_relaxed);
        nodeType.store(NodeType::LOSS, std::memory_order_release);
    }
    propagate_proof_to_parents();
}

void Node::mark_as_draw(int ply) {
    {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);
        if (nodeType.load(std::memory_order_relaxed) != NodeType::UNSOLVED) {
            return;
        }
        endInPly.store(ply, std::memory_order_relaxed);
        nodeType.store(NodeType::DRAW, std::memory_order_release);
    }
    propagate_proof_to_parents();
}

bool Node::update_child_node_type_from(
    int childIdx, const Node* expectedChild, NodeType childType) {
    if (!SearchParams::ENABLE_MCTS_SOLVER
        || childType == NodeType::UNSOLVED) {
        return false;
    }

    bool becameSolved = false;
    {
        std::unique_lock<std::shared_mutex> guard(nodeMutex);

        if (nodeType.load(std::memory_order_relaxed) != NodeType::UNSOLVED
            || childIdx < 0
            || static_cast<size_t>(childIdx) >= children.size()
            || (expectedChild && children[childIdx].get() != expectedChild)) {
            return false;
        }

        if (childNodeTypes.size() < children.size()) {
            const size_t oldSize = childNodeTypes.size();
            childNodeTypes.resize(children.size(), NodeType::UNSOLVED);
            unsolvedChildCount.fetch_add(
                static_cast<int>(children.size() - oldSize),
                std::memory_order_relaxed);
        }
        if (childNodeTypes[childIdx] != NodeType::UNSOLVED) {
            return false;
        }

        childNodeTypes[childIdx] = childType;
        unsolvedChildCount.fetch_sub(1, std::memory_order_relaxed);

        // Child values are from the child's perspective, so a losing child is
        // an immediately proven winning move for this node.
        if (childType == NodeType::LOSS) {
            const int childPly = children[childIdx]
                ? children[childIdx]->get_end_in_ply()
                : 0;
            endInPly.store(childPly + 1, std::memory_order_relaxed);
            nodeType.store(NodeType::WIN, std::memory_order_release);
            becameSolved = true;
        } else if (unsolvedChildCount.load(std::memory_order_relaxed) == 0
                   && m_is_expanded.load(std::memory_order_relaxed)
                   && !candidateGenerator.hasNext()) {
            // A loss/draw is proven only after every legal action is expanded.
            bool allWins = true;
            bool hasDrawn = false;
            int longestPly = 0;
            for (size_t index = 0; index < childNodeTypes.size(); ++index) {
                if (childNodeTypes[index] != NodeType::WIN) {
                    allWins = false;
                }
                if (childNodeTypes[index] == NodeType::DRAW) {
                    hasDrawn = true;
                }
                if (children[index]) {
                    longestPly = std::max(
                        longestPly, children[index]->get_end_in_ply());
                }
            }

            if (allWins) {
                endInPly.store(longestPly + 1, std::memory_order_relaxed);
                nodeType.store(NodeType::LOSS, std::memory_order_release);
                becameSolved = true;
            } else if (hasDrawn) {
                nodeType.store(NodeType::DRAW, std::memory_order_release);
                becameSolved = true;
            }
        }
    }

    if (becameSolved) {
        propagate_proof_to_parents();
    }
    return becameSolved;
}

void Node::initialize_root_gumbel_locked(
    const SearchParams::RuntimeConfig& config) {
    rootGumbelEnabled = config.enableGumbelRootSearch
        && m_depth.load(std::memory_order_relaxed) == 0;
    rootGumbelValueScale = config.rootGumbelValueScale;
    rootGumbelExpansionTarget = 0;
    rootGumbelRoundQuota = 1;
    rootGumbelActive.clear();
    rootGumbelWaiting.clear();
    childGumbelScores.clear();
    if (!rootGumbelEnabled) {
        return;
    }

    candidateGenerator.prepareGumbelPool(
        static_cast<size_t>(std::max(1, config.rootGumbelPoolSize)),
        config.rootNoiseSeed
            ^ positionHash.load(std::memory_order_relaxed)
            ^ 0xd1b54a32d192ed03ULL);
    rootGumbelExpansionTarget = std::max(
        1, config.rootGumbelInitialCandidates);
}

void Node::configure_root_search(
    const SearchParams::RuntimeConfig& config) {
    std::unique_lock<std::shared_mutex> guard(nodeMutex);
    if (!m_is_expanded.load(std::memory_order_relaxed)) {
        rootGumbelEnabled = config.enableGumbelRootSearch;
        return;
    }

    const std::vector<float> promotedPriors =
        candidateGenerator.reprepareJointPolicyPool(
            static_cast<size_t>(std::max(0, config.rootJointPolicyTopK)),
            config.jointPolicyResidualScale);
    if (promotedPriors.size() == childPriors.size()) {
        childPriors = promotedPriors;
    }

    rootGumbelEnabled = config.enableGumbelRootSearch;
    rootGumbelValueScale = config.rootGumbelValueScale;
    rootGumbelRoundQuota = 1;
    rootGumbelActive.clear();
    rootGumbelWaiting.clear();
    childGumbelScores.resize(children.size());
    if (!rootGumbelEnabled) {
        candidateGenerator.restoreFactorizedOrder();
        rootGumbelExpansionTarget = expandedCount;
        return;
    }

    const uint64_t seed = config.rootNoiseSeed
        ^ positionHash.load(std::memory_order_relaxed)
        ^ 0xd1b54a32d192ed03ULL;
    candidateGenerator.prepareGumbelPool(
        static_cast<size_t>(std::max(1, config.rootGumbelPoolSize)), seed);

    std::mt19937_64 randomEngine(seed ^ 0x94d049bb133111ebULL);
    std::uniform_real_distribution<double> uniform(
        std::nextafter(0.0, 1.0), std::nextafter(1.0, 0.0));
    std::vector<int> rankedChildren(children.size());
    std::iota(rankedChildren.begin(), rankedChildren.end(), 0);
    for (size_t index = 0; index < children.size(); ++index) {
        const double sample = uniform(randomEngine);
        const float gumbel = static_cast<float>(
            -std::log(-std::log(sample)));
        childGumbelScores[index] = std::log(
            std::max(childPriors[index], 1.0e-30f)) + gumbel;
    }
    std::sort(rankedChildren.begin(), rankedChildren.end(),
              [this](int lhs, int rhs) {
                  return childGumbelScores[lhs] > childGumbelScores[rhs];
              });

    const size_t tournamentCount = std::min(
        rankedChildren.size(), static_cast<size_t>(
            std::max(1, config.rootGumbelPoolSize)));
    const size_t initialCount = std::min(
        tournamentCount,
        static_cast<size_t>(std::max(1, config.rootGumbelInitialCandidates)));
    for (size_t rank = 0; rank < tournamentCount; ++rank) {
        const int childIdx = rankedChildren[rank];
        if (rank < initialCount) {
            rootGumbelActive.push_back({childIdx, childVisits[childIdx]});
        } else {
            rootGumbelWaiting.push_back(childIdx);
        }
    }
    rootGumbelExpansionTarget = std::max(
        expandedCount,
        std::max(1, config.rootGumbelInitialCandidates));
}

void Node::replenish_root_gumbel_locked(
    const SearchParams::RuntimeConfig& config) {
    int remaining = std::max(1, config.rootGumbelReplenishment);
    while (remaining > 0 && !rootGumbelWaiting.empty()) {
        const int childIdx = rootGumbelWaiting.front();
        rootGumbelWaiting.erase(rootGumbelWaiting.begin());
        if (childIdx >= 0 && static_cast<size_t>(childIdx) < childVisits.size()
            && children[childIdx]
            && children[childIdx]->get_node_type() != NodeType::WIN) {
            rootGumbelActive.push_back({childIdx, childVisits[childIdx]});
            --remaining;
        }
    }
    if (remaining > 0) {
        const int pendingCandidates = static_cast<int>(std::min(
            candidateGenerator.pendingGumbelCount(),
            static_cast<size_t>(remaining)));
        rootGumbelExpansionTarget += pendingCandidates;
        remaining -= pendingCandidates;
    }
    if (remaining > 0 && rootGumbelWaiting.empty()
        && candidateGenerator.pendingGumbelCount() == 0) {
        // The configured candidate pool is exhausted. Continuing to replenish
        // from the full Cartesian action space permanently starves tactical
        // depth, so hand the sampled candidates back to ordinary root PUCT.
        rootGumbelEnabled = false;
        rootGumbelActive.clear();
        rootGumbelExpansionTarget = expandedCount;
    }
}

void Node::advance_root_gumbel_round_locked(
    const SearchParams::RuntimeConfig& config) {
    std::erase_if(rootGumbelActive, [this](const RootGumbelEntry& entry) {
        return entry.childIdx < 0
            || static_cast<size_t>(entry.childIdx) >= children.size()
            || !children[entry.childIdx]
            // A WIN for the child is a solver-proven losing root action.
            || children[entry.childIdx]->get_node_type() == NodeType::WIN;
    });
    if (rootGumbelActive.empty()) {
        rootGumbelRoundQuota = 1;
        replenish_root_gumbel_locked(config);
        return;
    }

    const bool roundComplete = std::all_of(
        rootGumbelActive.begin(), rootGumbelActive.end(),
        [this](const RootGumbelEntry& entry) {
            return childVisits[entry.childIdx] - entry.baselineVisits
                >= rootGumbelRoundQuota;
        });
    if (!roundComplete) {
        return;
    }

    auto rankScore = [this, &config](const RootGumbelEntry& entry) {
        const int childIdx = entry.childIdx;
        const NodeType type = children[childIdx]->get_node_type();
        if (type == NodeType::LOSS) {
            return std::numeric_limits<float>::infinity();
        }
        const float q = type == NodeType::DRAW ? 0.0f : qValues[childIdx];
        return childGumbelScores[childIdx]
            + config.rootGumbelValueScale * q;
    };

    if (rootGumbelActive.size() > 1) {
        std::sort(rootGumbelActive.begin(), rootGumbelActive.end(),
                  [&rankScore](const RootGumbelEntry& lhs,
                               const RootGumbelEntry& rhs) {
                      return rankScore(lhs) > rankScore(rhs);
                  });
        rootGumbelActive.resize((rootGumbelActive.size() + 1) / 2);
        for (RootGumbelEntry& entry : rootGumbelActive) {
            entry.baselineVisits = childVisits[entry.childIdx];
        }
        rootGumbelRoundQuota = std::min(
            std::max(1, config.rootGumbelMaxRoundVisits),
            rootGumbelRoundQuota * 2);
        return;
    }

    rootGumbelActive.front().baselineVisits =
        childVisits[rootGumbelActive.front().childIdx];
    rootGumbelRoundQuota = 1;
    replenish_root_gumbel_locked(config);
    if (rootGumbelActive.size() == 1
        && rootGumbelExpansionTarget <= expandedCount) {
        rootGumbelRoundQuota = std::min(
            std::max(1, config.rootGumbelMaxRoundVisits), 2);
    }
}

Node::ChildSelection Node::select_child_and_apply_virtual_loss(
    const SearchParams::RuntimeConfig& config,
    const std::unordered_set<const Node*>* blockedNodes) {
    std::unique_lock<std::shared_mutex> guard(nodeMutex);
    
    // 1. Initial validation
    size_t numExpanded = static_cast<size_t>(expandedCount);
    if (numExpanded == 0 || children.empty()) {
        return {};
    }

    if (rootGumbelEnabled) {
        advance_root_gumbel_round_locked(config);
        if (expandedCount < rootGumbelExpansionTarget
            && candidateGenerator.pendingGumbelCount() > 0) {
            return {};
        }
    }

    // 2. Precompute constants for the selection loop (CrazyAra-aligned)
    int visits = m_visits.load(std::memory_order_relaxed) + virtualVisitSum;
    const float sqrtVisits = std::sqrt(static_cast<float>(visits));
    const float c = SearchParams::get_cpuct(
        static_cast<float>(visits), config.cpuctInit, config.cpuctBase);
    const float explorationBase = c * sqrtVisits;

    const size_t limit = std::min(numExpanded, children.size());

    float visitedPolicySum = 0.0f;
    if (config.enableDynamicFpu && visits > 0) {
        for (size_t i = 0; i < limit; i++) {
            if (childVisits[i] + virtualLoss[i] > 0) {
                visitedPolicySum += childPriors[i];
            }
        }
    }
    const float parentQ = visits > 0 ? (valueSum / static_cast<float>(visits)) : 0.0f;
    const float fpuQ = config.enableDynamicFpu && visits > 0
        ? std::clamp(parentQ - config.fpuReduction * std::sqrt(std::max(0.0f, visitedPolicySum)), -1.0f, 1.0f)
        : SearchParams::Q_INIT;

    const bool hasNonLosingAlternative = nodeType.load(std::memory_order_relaxed) == NodeType::UNSOLVED
        && std::any_of(children.begin(), children.begin() + limit,
            [](const std::shared_ptr<Node>& child) {
                return child && child->get_node_type() != NodeType::WIN;
            });
    std::vector<uint8_t> unavailableChildren;
    std::shared_ptr<Node> pendingEvaluation;
    std::vector<int> gumbelBaselines;
    if (rootGumbelEnabled) {
        gumbelBaselines.resize(limit, -1);
        for (const RootGumbelEntry& entry : rootGumbelActive) {
            if (entry.childIdx >= 0
                && static_cast<size_t>(entry.childIdx) < limit) {
                gumbelBaselines[entry.childIdx] = entry.baselineVisits;
            }
        }
    }

    while (true) {
        float bestScore = -std::numeric_limits<float>::infinity();
        int bestGumbelProgress = std::numeric_limits<int>::max();
        std::shared_ptr<Node> bestChild = nullptr;
        int selectedIdx = -1;

        // 4. Iterate only over expanded children
        for (size_t i = 0; i < limit; i++) {
            if (!children[i]
                || (blockedNodes && blockedNodes->contains(children[i].get()))
                || (!unavailableChildren.empty() && unavailableChildren[i])) {
                continue;
            }
            if (rootGumbelEnabled && gumbelBaselines[i] < 0) {
                continue;
            }
            if (hasNonLosingAlternative
                && children[i]->get_node_type() == NodeType::WIN) {
                continue;
            }

            const int vl_i = virtualLoss[i];
            const uint32_t n_i = static_cast<uint32_t>(childVisits[i]);
            const uint32_t n_effective = n_i + static_cast<uint32_t>(vl_i);
            
            float q_i;
            if (n_effective == 0) {
                q_i = fpuQ;
            } else if (vl_i == 0) {
                q_i = qValues[i];
            } else {
                const SearchParams::VirtualStyle style = SearchParams::get_virtual_style(n_i);
                if (style == SearchParams::VirtualStyle::VIRTUAL_LOSS) {
                    q_i = (childValueSum[i] - static_cast<float>(vl_i)) / static_cast<float>(n_effective);
                } else if (style == SearchParams::VirtualStyle::VIRTUAL_OFFSET) {
                    q_i = qValues[i] - static_cast<float>(vl_i) * static_cast<float>(SearchParams::VIRTUAL_OFFSET_STRENGTH);
                } else {
                    q_i = qValues[i];
                }
            }

            float score;
            int gumbelProgress = 0;
            if (rootGumbelEnabled) {
                const NodeType childType = children[i]->get_node_type();
                const float solverQ = childType == NodeType::LOSS
                    ? 1.0f
                    : childType == NodeType::DRAW ? 0.0f : q_i;
                score = childGumbelScores[i]
                    + config.rootGumbelValueScale * solverQ;
                gumbelProgress = static_cast<int>(n_effective)
                    - gumbelBaselines[i];
            } else {
                const float u_i = explorationBase * childPriors[i]
                    / (1.0f + static_cast<float>(n_effective));
                score = q_i + u_i;
            }

            const bool isBetter = rootGumbelEnabled
                ? (gumbelProgress < bestGumbelProgress
                   || (gumbelProgress == bestGumbelProgress
                       && score > bestScore))
                : score > bestScore;
            if (isBetter) {
                bestScore = score;
                bestGumbelProgress = gumbelProgress;
                bestChild = children[i];
                selectedIdx = static_cast<int>(i);
            }
        }

        if (selectedIdx < 0) {
            return {nullptr, -1, false, pendingEvaluation};
        }

        bool evaluationReserved = false;
        if (!bestChild->is_expanded()
            && bestChild->get_node_type() == NodeType::UNSOLVED) {
            if (!bestChild->try_reserve_evaluation()) {
                pendingEvaluation = bestChild;
                if (unavailableChildren.empty()) {
                    unavailableChildren.resize(limit, 0);
                }
                unavailableChildren[static_cast<size_t>(selectedIdx)] = 1;
                continue;
            }
            if (bestChild->is_expanded()
                || bestChild->get_node_type() != NodeType::UNSOLVED) {
                bestChild->release_evaluation_reservation();
            } else {
                evaluationReserved = true;
            }
        }

        virtualLoss[static_cast<size_t>(selectedIdx)]++;
        virtualVisitSum++;
        return {bestChild, selectedIdx, evaluationReserved, nullptr};
    }
}
