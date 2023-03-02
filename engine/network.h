#ifndef NETWORK_H 
#define NETWORK_H

#include "engine.h"
#include "bugboard.h"
#include "planes.h"
#include "constants.h"

class Network {
    private:
        int batchSize = 16; 
        Engine engine; 
        float* inputPlanes;
        float* valueOutput;
        float* policyOutput;

    public:
        Network(); 
        ~Network(); 
        void bind_input(Bugboard& board, Stockfish::Color side, int idx); 
        std::pair<std::vector<float*>, std::vector<float*>> forward(); 
        int get_batch_size() {
            return batchSize;
        }
};

#endif