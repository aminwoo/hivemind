#ifndef SEARCHTHREAD_H
#define SEARCHTHREAD_H

#include <chrono> 
#include <mutex>
#include <math.h>

#include "bugboard.h"
#include "node.h"
#include "network.h"
#include "searchinfo.h"

class Node;

class SearchThread {

    private: 
        Node* root; 
        SearchInfo* searchInfo;
        bool running = false; 
        std::map<std::string, Node*> transpositionTable; 
        std::vector<std::vector<Node*>> trajectoryBuffers; 

    public: 
        Network network; 
        SearchThread(); 
        ~SearchThread(); 
        Node* get_root_node(); 
        SearchInfo* get_search_info();
        void add_trajectory_buffer(); 
        void set_search_info(SearchInfo* info);
        void set_root_node(Node* node);
        void set_is_running(bool value); 
        void add_leaf_node(Bugboard& board, std::vector<Node*>& leafNodes, std::vector<Node*>& trajectoryBuffer); 
        void expand_leaf_node(Node* leaf, std::vector<std::pair<int, Stockfish::Move>> actions, DynamicVector<float> priors, Stockfish::Color action_side);
        void backup_leaf_node(Bugboard& board, float value, std::vector<Node*>& trajectoryBuffer);
        void run_iteration(std::vector<Bugboard>& boards);
        bool is_running(); 
};

void run_search_thread(SearchThread *t, Bugboard& board);

#endif