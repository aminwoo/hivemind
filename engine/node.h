#ifndef NODE_H
#define NODE_H

#include <math.h>

#include "Fairy-Stockfish/src/types.h"

#include "bugboard.h"
#include "constants.h"
#include "utils.h"

enum NodeType { 
    TERMINAL, 
    CAPTURE, 
    CHECK,
    NONE
};

class Node {
    private:
        Node* parent;
        std::vector<std::shared_ptr<Node>> children;  
        std::pair<int, Stockfish::Move> action; 
        int depth = 0; 
        int visits = 0; 
        float prior = 0.0f; 
        float total_value = -1.0f; 
        bool added = false; 
        bool expanded = false;
        Stockfish::Color action_side; 

    public:
        Node(Node* parent, Stockfish::Color action_side) : parent(parent), action_side(action_side) {};
        ~Node(); 

        //void set_action_side(Stockfish::Color value) {
        //    action_side = value; 
        //}

        Stockfish::Color get_action_side() {
            return action_side;  
        }


        std::vector<Node*> get_principle_variation();
        
        void apply_virtual_loss(float loss);

        void remove_virtual_loss(float loss); 

        Node* get_best_child(); 

        Node* get_parent() {
            return parent; 
        }

        void set_depth(int value) {
            depth = value; 
        }

        int get_depth() {
            return depth; 
        }

        bool get_added() {
            return added; 
        }

        void set_added(bool value) {
            added = value; 
        }

        void add_child(std::shared_ptr<Node> child) {
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

        void set_action(std::pair<int, Stockfish::Move> value) {
            action = value; 
        }

        int get_visits() {
            return visits; 
        }

        float get_prior() {
            return prior; 
        }
        float get_total_value() {
            return total_value; 
        }

        void increment_visits() {
            visits++; 
        }
        void set_prior(float value) {
            prior = value; 
        }
        void add_value(float value) {
            total_value += value; 
        }
        
        float Q() {
            return get_total_value() / (1 + get_visits());
        }

        float U() {
            return (sqrt(parent->get_visits()) * get_prior() / (1.0f + get_visits()));
        }
};

#endif