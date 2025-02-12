#pragma once

#include <vector>
#include <mutex>
#include <math.h>
#include <memory>

#include "Fairy-Stockfish/src/types.h"
#include "board.h"
#include "constants.h"
#include "utils.h"

/**
 * @brief Represents a node in the search tree used for evaluating moves.
 *
 * This class maintains statistical information (e.g., Q-values, visit counts)
 * and manages child nodes as part of a search algorithm.
 */
class Node {
    private:
        std::mutex mtx; ///< Mutex to ensure thread safety.

        std::vector<float> qValues;         ///< Q-values for each child node.
        std::vector<float> childValueSum;     ///< Cumulative value sum for each child.
        std::vector<float> childPriors;       ///< Prior probabilities for each child.
        std::vector<int> childVisits;         ///< Visit counts for each child.
        std::vector<std::shared_ptr<Node>> children; ///< Child nodes.
        std::pair<int, Stockfish::Move> action;///< Action that led to this node.
        int bestChildIdx = -1;                ///< Index of the best child.
        float valueSum = -1.0f;               ///< Cumulative value sum for this node.
        int m_depth = 0;                      ///< Depth of this node in the tree.
        int m_visits = 0;                     ///< Number of visits to this node.
        bool m_is_added = false;              ///< Flag indicating if node has been added.
        bool m_is_expanded = false;           ///< Flag indicating if node has been expanded.
        Stockfish::Color actionSide;          ///< The side (color) associated with the action.

    public:
        /**
         * @brief Constructs a Node with the specified action side.
         * @param actionSide The side associated with the action leading to this node.
         */
        Node(Stockfish::Color actionSide) : actionSide(actionSide) {};
        ~Node() = default;

        /**
         * @brief Updates the statistics for a child node.
         * @param childIdx Index of the child to update.
         * @param value Value to add.
         */
        void update(size_t childIdx, float value) {
            childValueSum[childIdx] += value; 
            childVisits[childIdx]++; 
            qValues[childIdx] = childValueSum[childIdx] / (1.0f + childVisits[childIdx]);
            valueSum += value; 
            m_visits++; 
        }

        /**
         * @brief Updates the node statistics for a terminal state.
         * @param value Terminal value to add.
         */
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

        /**
         * @brief Sets the depth of this node.
         * @param value Depth value.
         */
        void set_depth(int value) {
            m_depth = value; 
        }

        /**
         * @brief Returns the depth of this node.
         * @return int Depth value.
         */
        int get_depth() {
            return m_depth; 
        }

        /**
         * @brief Checks if this node has been added to the tree.
         * @return true if added, false otherwise.
         */
        bool is_added() {
            return m_is_added; 
        }

        /**
         * @brief Marks this node as added.
         * @param value Boolean indicating if the node is added.
         */
        void set_is_added(bool value) {
            m_is_added = value; 
        }

        /**
         * @brief Returns the index of the best child node.
         * @return int Best child's index.
         */
        int get_best_child_idx() {
            return bestChildIdx; 
        }

        /**
         * @brief Sets the index of the best child node.
         * @param value Best child's index.
         */
        void set_best_child_idx(int value) {
            bestChildIdx = value; 
        }

        /**
         * @brief Determines if this node represents a terminal state.
         * @return true if terminal, false otherwise.
         */
        bool is_terminal() const;

        /**
         * @brief Adds a child node with an associated action and prior probability.
         * @param child Shared pointer to the child node.
         * @param action Action associated with the child.
         * @param prior Prior probability for the child.
         */
        void add_child(std::shared_ptr<Node> child, std::pair<int, Stockfish::Move> action, float prior) {
            childValueSum.push_back(-1.0f);
            childPriors.push_back(prior);
            childVisits.push_back(0);
            child->set_action(action); 
            child->set_depth(this->get_depth() + 1);
            children.emplace_back(child); 
            qValues.push_back(-1.0f);
        }

        /**
         * @brief Returns the child nodes.
         * @return std::vector<std::shared_ptr<Node>> List of child nodes.
         */
        std::vector<std::shared_ptr<Node>> get_children() {
            return children;
        }

        /**
         * @brief Checks if the node has been expanded.
         * @return true if expanded, false otherwise.
         */
        bool get_is_expanded() {
            return m_is_expanded; 
        }

        /**
         * @brief Sets the expanded status of the node.
         * @param value Boolean indicating if the node is expanded.
         */
        void set_is_expanded(bool value) {
            m_is_expanded = value;
        }

        /**
         * @brief Sets the cumulative value for this node.
         * @param value Cumulative value.
         */
        void set_value(float value) {
            valueSum = value;
        }

        /**
         * @brief Returns the action associated with this node.
         * @return std::pair<int, Stockfish::Move> Action pair.
         */
        std::pair<int, Stockfish::Move> get_action() {
            return action; 
        }

        /**
         * @brief Returns the side associated with the action of this node.
         * @return Stockfish::Color Action side.
         */
        Stockfish::Color get_action_side() {
            return actionSide;  
        }

        /**
         * @brief Returns the action side of the best child node.
         * @return Stockfish::Color Action side.
         */
        Stockfish::Color get_child_action_side() {
            return children[bestChildIdx]->get_action_side(); 
        }

        /**
         * @brief Sets the action side for this node.
         * @param value Action side.
         */
        void set_action_side(Stockfish::Color value) {
            actionSide = value;  
        }

        /**
         * @brief Sets the action for this node.
         * @param value Action pair.
         */
        void set_action(std::pair<int, Stockfish::Move> value) {
            action = value; 
        }

        /**
         * @brief Returns the visit count for this node.
         * @return int Number of visits.
         */
        int get_visits() {
            return m_visits; 
        }

        /**
         * @brief Increments the visit count.
         */
        void increment_visits() {
            m_visits++; 
        }

        /**
         * @brief Decrements the visit count.
         */
        void decrement_visits() {
            m_visits--; 
        }

        /**
         * @brief Returns the cumulative value sum for this node.
         * @return float Value sum.
         */
        float get_value_sum() {
            return valueSum;
        }

        /**
         * @brief Calculates the average Q-value for this node.
         * @return float Q-value.
         */
        float Q() {
            return valueSum / (1.0f + m_visits);
        }
};

/**
 * @brief Computes a factor based on the number of visits.
 * @param visits Number of visits.
 * @return float Computed factor.
 */
float get_current_cput(float visits);
