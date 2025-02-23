#pragma once

#include <vector>
#include <mutex>
#include <math.h>
#include <memory>
#include <unordered_map>

#include "Fairy-Stockfish/src/types.h"
#include "board.h"
#include "constants.h"
#include "utils.h"

struct NodeKey {
    unsigned long key;
    Stockfish::Color color;
    bool hasNullMove;  

    bool operator==(const NodeKey &other) const {
        return key == other.key &&
               color == other.color &&
               hasNullMove == other.hasNullMove;
    }
};

namespace std {
    template <>
    struct hash<NodeKey> {
        std::size_t operator()(const NodeKey &k) const {
            std::size_t h1 = std::hash<unsigned long>()(k.key);
            std::size_t h2 = std::hash<int>()(static_cast<int>(k.color));
            std::size_t h3 = std::hash<bool>()(k.hasNullMove);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}



class Node;
using HashMap = std::unordered_map<NodeKey, std::weak_ptr<Node>>;

struct MapWithMutex {
    std::mutex mtx;
    HashMap hashTable;
    ~MapWithMutex() {}
};

class Node {
private:
    std::mutex mtx;
    unsigned long key;

    std::vector<std::shared_ptr<Node>> children;
    std::vector<std::pair<int, Stockfish::Move>> actions;
    std::vector<float> qValues;
    std::vector<float> childValueSum;
    std::vector<float> childPriors;
    std::vector<int> childVisits;
    int bestChildIdx = -1;

    float valueSum = -1.0f;
    int m_depth = 0;
    int m_visits = 0;
    bool m_is_added = false;
    bool m_is_expanded = false;

    Stockfish::Color actionSide;

public:
    Node(Stockfish::Color actionSide) : actionSide(actionSide) {}
    ~Node() = default;

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

    void set_hash_key(unsigned long key) {
        this->key = key;
    }

    unsigned long hash_key() {
        return key;
    }

    void lock();
    void unlock();

    void apply_virtual_loss_to_child(int childIdx);
    void revert_virtual_loss(int childIdx);

    std::vector<std::pair<int, Stockfish::Move>> get_principle_variation();

    std::shared_ptr<Node> get_child(int childIdx) {
        return children[childIdx];
    }

    std::shared_ptr<Node> get_best_child();

    bool has_null_move() {
        for (const auto& action : actions) {
            if (action.second == Stockfish::MOVE_NULL)
                return true;
        }
        return false; 
    }

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

    void set_child(int childIdx, std::shared_ptr<Node> child) {
        children[childIdx] = child;
    }

    Node* add_child(std::shared_ptr<Node> child, float prior) {
        childValueSum.push_back(-1.0f);
        childPriors.push_back(prior);
        childVisits.push_back(0);
        child->set_depth(get_depth() + 1);
        children.emplace_back(child);
        qValues.push_back(-1.0f);
        return child.get();
    }

    std::vector<std::shared_ptr<Node>> get_children() {
        return children;
    }

    bool is_expanded() {
        return m_is_expanded;
    }

    void set_is_expanded(bool value) {
        m_is_expanded = value;
    }

    void set_value(float value) {
        valueSum = value;
    }

    void set_actions(std::vector<std::pair<int, Stockfish::Move>> actions) {
        this->actions = actions;
    }

    std::pair<int, Stockfish::Move> get_action(int childIdx) {
        return actions[childIdx];
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
