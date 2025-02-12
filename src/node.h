#ifndef NODE_H
#define NODE_H

#include <vector>
#include <mutex>
#include <math.h>

#include "Fairy-Stockfish/src/types.h"

#include "board.h"
#include "constants.h"
#include "utils.h"


class Node {
    private:
        std::mutex mtx;

        std::vector<float> qValues;
        std::vector<float> childValueSum;
        std::vector<float> childPriors;
        std::vector<int> childVisits; 
        std::vector<std::shared_ptr<Node>> children;  
        std::pair<int, Stockfish::Move> action; 
        int bestChildIdx = -1; 
        float valueSum = -1.0f; 
        int m_depth = 0; 
        int m_visits = 0; 
        bool m_is_added = false; 
        bool m_is_expanded = false;
        Stockfish::Color actionSide; 

    public:
        Node(Stockfish::Color actionSide) : actionSide(actionSide) {};
        ~Node() {}; 

        void update(size_t childIdx, float value) {
            childValueSum[childIdx] += value; 
            childVisits[childIdx]++; 
            qValues[childIdx] = childValueSum[childIdx] / (1.0f + childVisits[childIdx]);
            valueSum += value; 
            m_visits++; 
        }

        void update_terminal(float value) {
            valueSum += value; 
            m_visits++; 
        }

        void lock();
        void unlock();

        void apply_virtual_loss_to_child(int childIdx);

        void revert_virtual_loss(int childIdx); 

        std::vector<Node*> get_principle_variation();
        
        Node* get_best_child(); 

        void set_depth(int value) {
            m_depth = value; 
        }

        int get_depth() {
            return m_depth; 
        }

        bool is_added() {
            return m_is_added; 
        }

        void set_is_added(bool value) {
            m_is_added = value; 
        }

        int get_best_child_idx() {
            return bestChildIdx; 
        }

        void set_best_child_idx(int value) {
            bestChildIdx = value; 
        }

        bool is_terminal() const;

        void add_child(std::shared_ptr<Node> child, std::pair<int, Stockfish::Move> action, float prior) {
            childValueSum.push_back(-1.0f);
            childPriors.push_back(prior);
            childVisits.push_back(0);
            child->set_action(action); 
            child->set_depth(this->get_depth() + 1);
            children.emplace_back(child); 
            qValues.push_back(-1.0f);
        }

        std::vector<std::shared_ptr<Node>> get_children() {
            return children;
        }

        bool get_is_expanded() {
            return m_is_expanded; 
        }

        void set_is_expanded(bool value) {
            m_is_expanded = value;
        }

        void set_value(float value) {
            valueSum = value;
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
            return m_visits; 
        }

        void increment_visits() {
            m_visits++; 
        }

        void decrement_visits() {
            m_visits--; 
        }

        float get_value_sum() {
            return valueSum;
        }

        float Q() {
            return valueSum / (1.0f + m_visits);
        }
};

float get_current_cput(float visits);

#endif
