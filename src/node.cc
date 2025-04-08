#include "node.h"
#include <limits>
#include <algorithm>

const double VIRTUAL_LOSS = 0.0; // Tunable parameter

void Node::apply_virtual_loss_to_child(int childIdx) {
    assert (m_is_expanded); 
    assert (children.size() > 0);
    assert (childVisits.size() > 0); 
    assert (childIdx < childVisits.size()); 
    ++childVisits[childIdx];
    ++m_visits;

    // Assume each child has a totalValue that affects selection.
    childValueSum[childIdx] -= VIRTUAL_LOSS;
    qValues[childIdx] = childValueSum[childIdx] / (1.0f + childVisits[childIdx]);
}

void Node::revert_virtual_loss(int childIdx) {
    --childVisits[childIdx];
    --m_visits;

    childValueSum[childIdx] += VIRTUAL_LOSS;
    qValues[childIdx] = childValueSum[childIdx] / (1.0f + childVisits[childIdx]);
}

std::shared_ptr<Node> Node::get_best_child() {
    std::shared_ptr<Node> bestChild = nullptr; 
    float bestValue = -std::numeric_limits<float>::infinity();
    float c = get_current_cput(m_visits); 
    size_t numChildren = children.size(); 
    std::vector<float> uValues(numChildren);

    // Calculate uValues = sqrt(visits) * childPriors / (1.0f + childVisits)
    float sqrtVisits = std::sqrt(m_visits);
    std::transform(childPriors.begin(), childPriors.end(),
                   childVisits.begin(),
                   uValues.begin(),
                   [sqrtVisits](float prior, float childVisit) -> float {
                       return sqrtVisits * prior / (1.0f + childVisit);
                   });

    for (size_t i = 0; i < numChildren; i++) {
        float value = qValues[i] + c * uValues[i];
        if (value > bestValue) {
            bestValue = value; 
            bestChild = children[i];  
            bestChildIdx = i;
        }
    }
    return bestChild;
}


std::vector<std::pair<int, Stockfish::Move>> Node::get_principle_variation() {
    std::vector<std::pair<int, Stockfish::Move>> pv;
    Node* currentNode = this;
    
    // Tunable parameters for Q-value integration.
    const double Qfactor = 0.25;  // weight for Q-value component (0 means only visits)
    const double Qthresh = 0.33; // if visits < Qthresh * maxVisits, ignore Q-value

    while (currentNode != nullptr) {
        currentNode->lock();

        auto children = currentNode->get_children();
        if (children.empty()) {
            currentNode->unlock();
            break;
        }

        // Compute total visits and maximum visits among the children.
        double sumVisits = 0.0;
        int maxVisits = 0;
        for (auto& child : children) {
            int visits = child->get_visits();
            sumVisits += visits;
            if (visits > maxVisits)
                maxVisits = visits;
        }
        
        // Select the child with the best combined score.
        Node* bestChild = nullptr;
        int bestChildIdx = -1; 
        double bestScore = -std::numeric_limits<double>::infinity();
        for (size_t childIdx = 0; childIdx < children.size(); childIdx++) {
            std::shared_ptr<Node> child = currentNode->get_child(childIdx); 
            
            int visits = child->get_visits();
            // Normalized visits component:
            double normalizedVisits = (sumVisits > 0) ? (static_cast<double>(visits) / sumVisits) : 0.0;

            // Rescale Q-value from [-1,1] to [0,1]
            double qValue = child->Q();
            double qScaled = (qValue + 1.0) / 2.0;
            // If the child's visits are too low, ignore its Q-value.
            if (visits < static_cast<int>(Qthresh * maxVisits))
                qScaled = 0.0;
            
            double score = (1.0 - Qfactor) * normalizedVisits + Qfactor * qScaled;
            if (score > bestScore) {
                bestScore = score;
                bestChild = child.get();
                bestChildIdx = childIdx; 
            }
        }

        if (bestChild == nullptr) {
            currentNode->unlock();
            break;
        }

        pv.push_back(currentNode->get_action(bestChildIdx));
        currentNode->unlock();
        currentNode = bestChild;
    }
    return pv;
}

void Node::lock() {
    mtx.lock();
}

void Node::unlock() {
    mtx.unlock();
}


float get_current_cput(float visits) {
    float cpuctInit = 2.5f; 
    float cpuctBase = 19652.0f; 
    return log((visits + cpuctBase + 1) / cpuctBase) + cpuctInit;
}