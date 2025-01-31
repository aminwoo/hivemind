#include "node.h"
#include <limits>
#include <algorithm>

Node* Node::get_best_child() {
    Node* bestChild = nullptr; 
    float bestValue = -std::numeric_limits<float>::infinity();
    float c = 2.5; 
    assert(visits >= 0); 

    size_t numChildren = children.size(); 

    std::vector<float> qValues(numChildren);
    std::vector<float> uValues(numChildren);

    // Calculate qValues = childValueSum / (1.0f + childVisits)
    std::transform(childValueSum.begin(), childValueSum.end(),
                   childVisits.begin(),
                   qValues.begin(),
                   [](float valueSum, float visit) -> float {
                       return valueSum / (1.0f + visit);
                   });

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
    Node* curr = this; 
    for (int i = 0; i < 10; i++) {
        if (curr->get_children().empty()) {
            break; 
        }
        Node* mostVisitedChild; 
        int mostVisits = 0; 
        for (auto& child: curr->get_children()) {
            if (child->get_visits() > mostVisits) {
                mostVisits = child->get_visits(); 
                mostVisitedChild = child.get(); 
            }
        }
        pv.emplace_back(mostVisitedChild); 
        curr = mostVisitedChild;
    }
    return pv; 
}
