#include "network.h"

Network::Network() : engine(batchSize) {
    inputPlanes = new float[batchSize * NUM_BUGHOUSE_VALUES()];
    valueOutput = new float[batchSize];
    policyOutput = new float[batchSize * 2 * NUM_POLICY_VALUES];
    engine.build("model.onnx", "trt_" + std::to_string(batchSize) + ".engine");
    engine.loadNetwork();
    engine.bind_executor_input(NUM_BUGHOUSE_VALUES()); 
    engine.bind_executor_value(1); 
    engine.bind_executor_policy(2 * NUM_POLICY_VALUES); 
    engine.predict(inputPlanes, valueOutput, policyOutput); // Throwaway predict since first predict is always slow */
}

Network::~Network() {
    
}

void Network::bind_input(Bugboard& board, Stockfish::Color side, int idx) {
    board_to_planes(board, inputPlanes + idx * NUM_BUGHOUSE_VALUES(), side); 
}

std::pair<std::vector<float*>, std::vector<float*>> Network::forward() {
    engine.predict(inputPlanes, valueOutput, policyOutput); 
    std::vector<float*> valueOutputs, policyOutputs;
    for (int i = 0; i < batchSize; i++) {
        valueOutputs.emplace_back(valueOutput + i); 
        policyOutputs.emplace_back(policyOutput + i * 2 * NUM_POLICY_VALUES);
    } 
    return {valueOutputs, policyOutputs}; 
}
