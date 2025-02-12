#include "node.h"
#include <limits>
#include <algorithm>

void Node::apply_virtual_loss_to_child(int childIdx) {
    assert (m_is_expanded); 
    assert (children.size() > 0);
    assert (childVisits.size() > 0); 
    assert (childIdx < childVisits.size()); 
    qValues[childIdx] = (double(qValues[childIdx]) * childVisits[childIdx] - 1) / double(childVisits[childIdx] + 1);
    ++childVisits[childIdx];
    ++m_visits;
}

void Node::revert_virtual_loss(int childIdx) {
    qValues[childIdx] = (double(qValues[childIdx]) * childVisits[childIdx] + 1) / (childVisits[childIdx] - 1);
    --childVisits[childIdx];
    --m_visits;
}

Node* Node::get_best_child() {
    Node* bestChild = nullptr; 
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
            bestChild = children[i].get(); 
            bestChildIdx = i;
        }
    }
    return bestChild;
}

std::vector<Node*> Node::get_principle_variation() {
    std::vector<Node*> pv; 
    Node* currentNode = this; 
    while (currentNode != nullptr) {
        currentNode->lock();

        Node* mostVisitedChild = nullptr; 
        int mostVisits = 0; 
        for (auto& child: currentNode->get_children()) {
            if (child->get_visits() > mostVisits) {
                mostVisits = child->get_visits(); 
                mostVisitedChild = child.get(); 
            }
        }
        if (mostVisitedChild == nullptr) {
            currentNode->unlock();
            break; 
        }

        pv.emplace_back(mostVisitedChild); 
        currentNode->unlock();
        currentNode = mostVisitedChild;
    }
    return pv; 
}

void Node::lock()
{
    mtx.lock();
}

void Node::unlock()
{
    mtx.unlock();
}


float get_current_cput(float visits)
{
    float cpuctInit = 2.5f; 
    float cpuctBase = 19652.0f; 
    return log((visits + cpuctBase + 1) / cpuctBase) + cpuctInit;
}