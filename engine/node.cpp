#include "node.h"

#include <limits>

void Node::apply_virtual_loss_to_child(int childIdx, float loss) {
    //assert (childVirtualLoss.size() > 0); 
    //std::cout << childVirtualLoss.size() << ' ' << childVisits.size() << std::endl;
    //childVirtualLoss[childIdx] += loss; 
    //childVisits[childIdx] += loss;
    assert (expanded); 
    assert (children.size() > 0);
    assert (childVisits.size() > 0); 
    assert (childIdx < childVisits.size()); 
    //childVirtualLoss[childIdx] += loss; 
    //childValueSum[childIdx] -= loss; 
    //childVisits[childIdx] += loss;
    //visits += loss;
    //std::cout << "Applying " << childIdx << std::endl; 
}

void Node::revert_virtual_loss_to_child(int childIdx, float loss) {
    //childVirtualLoss[childIdx] -= loss; 
    //childValueSum[childIdx] += loss; 
    //childVisits[childIdx] = std::max(childVisits[childIdx] - loss, 0.0f);
    //visits -= loss; 
    //visits = std::max(visits - (int)loss, 0);
    //std::cout << "Removing " << childIdx << std::endl; 
}

Node* Node::get_best_child() {
    Node* bestChild = nullptr; 
    float bestValue = -std::numeric_limits<float>::infinity();
    float c = 2.5; 
    assert(visits >= 0); 

    //std::cout << childVirtualLoss << std::endl; 
    //DynamicVector<float> qValues = ((childValueSum / (1.0f + childVisits)) * childVisits - childVirtualLoss) / (childVisits + childVirtualLoss); 
    DynamicVector<float> qValues = childValueSum / (1.0f + childVisits); 
    DynamicVector<float> uValues = sqrt(visits) * childPriors / (1.0f + childVisits);

    size_t numChildren = children.size(); 
    for (size_t i = 0; i < numChildren; i++) {
        float val = qValues[i] + c * uValues[i];
        if (val > bestValue) {
            bestValue = val; 
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