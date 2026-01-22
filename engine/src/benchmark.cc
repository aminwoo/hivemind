#include "benchmark.h"
#include "constants.h"
#include <iostream>
#include <chrono>
#include <cstdlib>

using namespace std;

void benchmark_inference(Engine& engine, int iterations) {
    // Allocate buffers
    float* obs = new float[BATCH_SIZE * NB_INPUT_VALUES()];
    float* value = new float[BATCH_SIZE];
    float* piA = new float[BATCH_SIZE * NB_POLICY_VALUES()];
    float* piB = new float[BATCH_SIZE * NB_POLICY_VALUES()];
    
    // Initialize with random data
    for (int i = 0; i < BATCH_SIZE * NB_INPUT_VALUES(); i++) {
        obs[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        engine.runInference(obs, value, piA, piB);
    }
    
    // Benchmark
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        engine.runInference(obs, value, piA, piB);
    }
    auto end = chrono::high_resolution_clock::now();
    
    double total_ms = chrono::duration<double, milli>(end - start).count();
    double avg_ms = total_ms / iterations;
    double inferences_per_sec = 1000.0 / avg_ms;
    
    cout << "=== Inference Benchmark ===" << endl;
    cout << "Iterations: " << iterations << endl;
    cout << "Total time: " << total_ms << " ms" << endl;
    cout << "Average time per inference: " << avg_ms << " ms" << endl;
    cout << "Inferences per second: " << inferences_per_sec << endl;
    cout << "===========================" << endl;
    
    delete[] obs;
    delete[] value;
    delete[] piA;
    delete[] piB;
}
