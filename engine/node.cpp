#include "node.h"

Node::~Node() {

}

void Node::apply_virtual_loss(float loss) {
    add_value(-loss); 
}

void Node::remove_virtual_loss(float loss) {
    add_value(loss); 
}

Node* Node::get_best_child() {
    Node* bestChild = nullptr; 
    float bestValue = -999; 
    float c = 2.5; 

    for (auto& child: get_children()) {
        float val = child->Q() + c * child->U();
        if (val > bestValue) {
            bestValue = val; 
            bestChild = child.get(); 
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