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

    // 2. Precompute constants for the selection loop
    const float sqrtVisits = std::sqrt(static_cast<float>(m_visits));
    const float c = SearchParams::get_cpuct(m_visits);
    const float explorationBase = c * sqrtVisits;

    // 3. Parent-Relative FPU Calculation
    // Sets the 'default' value for nodes with 0 visits
    float parentQ = (m_visits > 0) ? (valueSum / static_cast<float>(m_visits)) : 0.0f;
    const float fpuValue = std::max(-1.0f, parentQ - SearchParams::FPU_REDUCTION);

    float bestScore = -std::numeric_limits<float>::infinity();
    std::shared_ptr<Node> bestChild = nullptr;
    int selectedIdx = -1;

    // 4. Iterate only over expanded children
    size_t limit = std::min(numExpanded, children.size());
    for (size_t i = 0; i < limit; i++) {
        float n_i = static_cast<float>(childVisits[i]);
        float vl_i = static_cast<float>(virtualLoss[i]);  // Virtual loss for this child
        
        // Effective visit count includes virtual loss
        float n_effective = n_i + vl_i;
        
        // Q-value with virtual loss: treat virtual visits as losses (-1)
        // Q = (sum - virtual_loss) / (visits + virtual_loss)
        float q_i;
        if (n_effective == 0.0f) {
            q_i = fpuValue;
        } else {
            q_i = (childValueSum[i] - vl_i) / n_effective;
        }

        // PUCT formula: Score = Q + C * P * (sqrt(N_parent) / (1 + n_effective))
        float u_i = explorationBase * childPriors[i] / (1.0f + n_effective);
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