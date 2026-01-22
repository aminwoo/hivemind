#include "node.h"
#include <limits>
#include <algorithm>

std::shared_ptr<Node> Node::get_best_expanded_child() {
    // 1. Initial validation
    size_t numExpanded = static_cast<size_t>(expandedCount);
    if (numExpanded == 0 || children.empty()) {
        return nullptr;
    }

    // 2. Precompute constants for the selection loop
    const float sqrtVisits = std::sqrt(static_cast<float>(m_visits));
    const float c = get_current_cput(m_visits);
    const float explorationBase = c * sqrtVisits;

    // 3. Parent-Relative FPU Calculation
    // Sets the 'default' value for nodes with 0 visits
    float parentQ = (m_visits > 0) ? (valueSum / static_cast<float>(m_visits)) : 0.0f;
    const float fpuReduction = 0.4f; 
    const float fpuValue = std::max(-1.0f, parentQ - fpuReduction);

    float bestScore = -std::numeric_limits<float>::infinity();
    std::shared_ptr<Node> bestChild = nullptr;

    // 4. Iterate only over expanded children
    size_t limit = std::min(numExpanded, children.size());
    for (size_t i = 0; i < limit; i++) {
        float n_i = static_cast<float>(childVisits[i]);
        
        // Determine Q-value: Use FPU if the child hasn't been visited yet
        float q_i = (n_i == 0.0f) ? fpuValue : (childValueSum[i] / n_i);

        // PUCT formula: Score = Q + C * P * (sqrt(N_parent) / (1 + n_child))
        float u_i = explorationBase * childPriors[i] / (1.0f + n_i);
        float score = q_i + u_i;

        if (score > bestScore) {
            bestScore = score;
            bestChild = children[i];
            bestChildIdx = static_cast<int>(i); // Update index for backprop
        }
    }

    return bestChild;
}

float get_current_cput(float totalVisits) {
    const float cpuctInit = 2.5f; 
    const float cpuctBase = 19652.0f; 
    
    return std::log((totalVisits + cpuctBase + 1.0f) / cpuctBase) + cpuctInit;
}