#include "node.h"
#include "search_params.h"
#include <limits>
#include <algorithm>

std::shared_ptr<Node> Node::get_best_expanded_child() {
    auto [child, idx] = get_best_expanded_child_with_idx();
    return child;
}

std::pair<std::shared_ptr<Node>, int> Node::get_best_expanded_child_with_idx() {
    std::lock_guard<std::mutex> guard(nodeMutex);
    
    // 1. Initial validation
    size_t numExpanded = static_cast<size_t>(expandedCount);
    if (numExpanded == 0 || children.empty()) {
        return {nullptr, -1};
    }

    // 2. Precompute constants for the selection loop (CrazyAra-aligned)
    const float sqrtVisits = std::sqrt(static_cast<float>(m_visits));
    const float c = SearchParams::get_cpuct(static_cast<float>(m_visits));
    const float explorationBase = c * sqrtVisits;

    // 3. Parent-Relative FPU Calculation (Lc0/CrazyAra-style)
    // Sets the 'default' value for nodes with 0 visits
    float parentQ = (m_visits > 0) ? (valueSum / static_cast<float>(m_visits)) : 0.0f;
    const float fpuValue = std::max(-1.0f, parentQ - SearchParams::FPU_REDUCTION);

    float bestScore = -std::numeric_limits<float>::infinity();
    std::shared_ptr<Node> bestChild = nullptr;
    int selectedIdx = -1;

    // 4. Iterate only over expanded children
    size_t limit = std::min(numExpanded, children.size());
    for (size_t i = 0; i < limit; i++) {
        uint32_t n_i = static_cast<uint32_t>(childVisits[i]);
        int vl_i = virtualLoss[i];  // Virtual loss counter for this child
        
        // Effective visit count includes virtual loss visits
        uint32_t n_effective = n_i + static_cast<uint32_t>(vl_i);
        
        // Get Q-value with virtual loss applied based on configured style
        // CrazyAra uses qValues array which stores the modified Q-value
        float q_i;
        if (n_effective == 0) {
            q_i = fpuValue;
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
    
    bestChildIdx = selectedIdx;  // Update for backward compatibility
    return {bestChild, selectedIdx};
}