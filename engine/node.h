#ifndef NODE_H
#define NODE_H

#include <math.h>

#include "Fairy-Stockfish/src/types.h"

#include "bugboard.h"
#include "constants.h"
#include "utils.h"

class Node {
    private:
        DynamicVector<float> childVirtualLoss;
        DynamicVector<float> childValueSum;
        DynamicVector<float> childPriors;
        DynamicVector<int> childVisits; 
        std::vector<std::shared_ptr<Node>> children;  
        std::pair<int, Stockfish::Move> action; 
        int bestChildIdx = -1; 
        float valueSum = -1.0f; 
        int visits = 0; 
        bool added = false; 
        bool expanded = false;
        Stockfish::Color actionSide; 

    public:
        Node(Stockfish::Color actionSide) : actionSide(actionSide) {};
        ~Node() {}; 

        void update(size_t childIdx, float value, Stockfish::Color childActionSide, Stockfish::Color actionSide) {
            childValueSum[childIdx] += value; 
            childVisits[childIdx]++; 
            if (actionSide != childActionSide) {
                value = -value; 
            }
            valueSum += value; 
            visits++; 
        }

        void update_terminal(float value) {
            valueSum += value; 
            visits++; 
        }

        std::vector<Node*> get_principle_variation();
        
        void apply_virtual_loss_to_child(int childIdx, float loss);

        void revert_virtual_loss_to_child(int childIdx, float loss); 

        Node* get_best_child(); 

        bool is_added() {
            return added; 
        }

        void set_added(bool value) {
            added = value; 
        }

        int get_idx() {
            return bestChildIdx; 
        }

        void set_idx(int value) {
            bestChildIdx = value; 
        }

        void add_child(std::shared_ptr<Node> child, std::pair<int, Stockfish::Move> action, float prior) {
            append(childVirtualLoss, 0.0f);
            append(childValueSum, -1.0f);
            append(childPriors, prior);
            append(childVisits, 0);
            child->set_action(action); 
            children.emplace_back(child); 
        }

        std::vector<std::shared_ptr<Node>> get_children() {
            return children;
        }

        bool get_expanded() {
            return expanded; 
        }

        void set_expanded(bool value) {
            expanded = value;
        }

        std::pair<int, Stockfish::Move> get_action() {
            return action; 
        }

        Stockfish::Color get_action_side() {
            return actionSide;  
        }

        Stockfish::Color get_child_action_side() {
            return children[bestChildIdx]->get_action_side(); 
        }

        void set_action_side(Stockfish::Color value) {
            actionSide = value;  
        }

        void set_action(std::pair<int, Stockfish::Move> value) {
            action = value; 
        }

        int get_visits() {
            return visits; 
        }

        void increment_visits() {
            visits++; 
        }

        void decrement_visits() {
            visits--; 
        }

        float Q() {
            return valueSum / (1.0f + visits);
        }
};

#endif