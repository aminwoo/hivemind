#include "node.h"
#include <limits>
#include <algorithm>

std::shared_ptr<Node> Node::get_best_child() {
    std::shared_ptr<Node> bestChild = nullptr; 
    float bestScore = -std::numeric_limits<float>::infinity();
    
    // CPUCT balances the SL Policy (P) vs the calculated Value (Q)
    float c = get_current_cput(m_visits); 

    // Common terms for the PUCT formula
    float sqrtVisits = std::sqrt(static_cast<float>(m_visits));

    /**
     * Parent-Relative FPU Calculation
     * We calculate the 'average value' of moves already searched at this node.
     * We then subtract a reduction (0.4f) to set a 'bar' for unvisited moves.
     */
    float parentQ = (m_visits > 0) ? (valueSum / m_visits) : 0.0f;
    const float fpuReduction = 0.4f; 
    float fpuValue = std::max(-1.0f, parentQ - fpuReduction);

    for (size_t i = 0; i < children.size(); i++) {
        float n_i = static_cast<float>(childVisits[i]);
        float q_i;

        // 1. Determine Q-value (Use FPU for unvisited nodes)
        if (n_i == 0) {
            q_i = fpuValue;
        } else {
            // Standard Q = W / N
            q_i = childValueSum[i] / n_i;
        }

        // 2. Calculate U-value (The exploration/Policy term)
        // (1.0f + n_i) ensures we don't divide by zero and stabilizes the score
        float u_i = c * childPriors[i] * (sqrtVisits / (1.0f + n_i));

        // 3. Combine for final PUCT score
        float score = q_i + u_i;

        if (score > bestScore) {
            bestScore = score; 
            bestChild = children[i];  
            bestChildIdx = i; // Store for backpropagation indexing
        }
    }
    
    return bestChild;
}

std::shared_ptr<Node> Node::get_best_expanded_child() {
    // Only consider children that have already been expanded
    size_t numExpanded = static_cast<size_t>(expandedCount);
    if (numExpanded == 0 || children.empty()) {
        return nullptr;
    }
    
    std::shared_ptr<Node> bestChild = nullptr;
    float bestValue = -std::numeric_limits<float>::infinity();
    float c = get_current_cput(m_visits);
    float sqrtVisits = std::sqrt(static_cast<float>(m_visits));
    
    // Only iterate over expanded children
    size_t limit = std::min(numExpanded, children.size());
    for (size_t i = 0; i < limit; i++) {
        float prior = childPriors[i];
        float visits = static_cast<float>(childVisits[i]);
        float uValue = sqrtVisits * prior / (1.0f + visits);
        float value = qValues[i] + c * uValue;
        
        if (value > bestValue) {
            bestValue = value;
            bestChild = children[i];
            bestChildIdx = static_cast<int>(i);
        }
    }
    return bestChild;
}

float get_current_cput(float visits) {
    float cpuctInit = 2.5f; 
    float cpuctBase = 19652.0f; 
    return log((visits + cpuctBase + 1) / cpuctBase) + cpuctInit;
}