#include "node.h"
#include "search_params.h"
#include <limits>
#include <algorithm>

Node::ChildSelection Node::select_child_and_apply_virtual_loss(
    const SearchParams::RuntimeConfig& config) {
    std::unique_lock<std::shared_mutex> guard(nodeMutex);
    
    // 1. Initial validation
    size_t numExpanded = static_cast<size_t>(expandedCount);
    if (numExpanded == 0 || children.empty()) {
        return {};
    }

    // 2. Precompute constants for the selection loop (CrazyAra-aligned)
    int visits = m_visits.load(std::memory_order_relaxed) + virtualVisitSum;
    const float sqrtVisits = std::sqrt(static_cast<float>(visits));
    const float c = SearchParams::get_cpuct(
        static_cast<float>(visits), config.cpuctInit, config.cpuctBase);
    const float explorationBase = c * sqrtVisits;

    const size_t limit = std::min(numExpanded, children.size());
    const bool hasNonLosingAlternative = nodeType == NodeType::UNSOLVED
        && std::any_of(children.begin(), children.begin() + limit,
            [](const std::shared_ptr<Node>& child) {
                return child && child->get_node_type() != NodeType::WIN;
            });
    std::vector<uint8_t> unavailableChildren;
    std::shared_ptr<Node> pendingEvaluation;

    while (true) {
        float bestScore = -std::numeric_limits<float>::infinity();
        std::shared_ptr<Node> bestChild = nullptr;
        int selectedIdx = -1;

        // 4. Iterate only over expanded children
        for (size_t i = 0; i < limit; i++) {
            if (!children[i]
                || (!unavailableChildren.empty() && unavailableChildren[i])) {
                continue;
            }
            if (hasNonLosingAlternative
                && children[i]->get_node_type() == NodeType::WIN) {
                continue;
            }

            uint32_t n_i = static_cast<uint32_t>(childVisits[i]);
            int vl_i = virtualLoss[i];  // Virtual loss counter for this child

            // Effective visit count includes virtual loss visits
            uint32_t n_effective = n_i + static_cast<uint32_t>(vl_i);
            
            // Get Q-value with virtual loss applied based on configured style
            // CrazyAra uses qValues array which stores the modified Q-value
            float q_i;
            if (n_effective == 0) {
                q_i = SearchParams::Q_INIT;
            } else {
                // Get base Q-value and apply virtual loss effect based on style
                SearchParams::VirtualStyle style = SearchParams::get_virtual_style(n_i);

                if (style == SearchParams::VirtualStyle::VIRTUAL_LOSS) {
                    // VIRTUAL_LOSS: Q = (sum - virtual_loss) / (visits + virtual_loss)
                    // This treats each virtual visit as a loss (-1 value)
                    q_i = (childValueSum[i] - static_cast<float>(vl_i)) / static_cast<float>(n_effective);
                } else if (style == SearchParams::VirtualStyle::VIRTUAL_OFFSET) {
                    // VIRTUAL_OFFSET: Q = Q - offset * virtual_loss_count
                    q_i = qValues[i] - static_cast<float>(vl_i) * static_cast<float>(SearchParams::VIRTUAL_OFFSET_STRENGTH);
                } else {
                    // VIRTUAL_VISIT: Q unchanged, only visit count increases
                    q_i = qValues[i];
                }
            }

            // PUCT formula (same as CrazyAra): Score = Q + C * P * (sqrt(N_parent) / (1 + n_effective))
            float u_i = explorationBase * childPriors[i] / (1.0f + static_cast<float>(n_effective));
            float score = q_i + u_i;

            if (score > bestScore) {
                bestScore = score;
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