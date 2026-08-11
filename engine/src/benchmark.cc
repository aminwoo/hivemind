#include "benchmark.h"

#include <chrono>
#include <cstdlib>
#include <iostream>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "board.h"
#include "constants.h"

using namespace std;

void benchmark_inference(Engine& engine, int iterations) {
    const int batchSize = engine.getBatchSize();
    const size_t inputElements = batchSize * NB_INPUT_VALUES();
    __half* obs = nullptr;
    if (cudaMallocHost(
            reinterpret_cast<void**>(&obs), inputElements * sizeof(__half))
        != cudaSuccess) {
        cerr << "Failed to allocate pinned inference benchmark input" << endl;
        return;
    }
    Engine::HalfInferenceOutputs outputs;
    
    // Initialize with random data
    for (size_t i = 0; i < inputElements; i++) {
        obs[i] = __float2half_rn(static_cast<float>(rand()) / RAND_MAX);
    }
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        engine.runInferenceHalf(obs, outputs);
    }
    
    // Benchmark
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        engine.runInferenceHalf(obs, outputs);
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
    cout << "Positions per second: " << inferences_per_sec * batchSize << endl;
    cout << "===========================" << endl;
    
    cudaFreeHost(obs);
}

static long long perft(Board& board, int depth) {
    if (depth == 0) return 1;
    
    auto movesA = board.legal_moves(BOARD_A);
    auto movesB = board.legal_moves(BOARD_B);
    
    if (depth == 1) return movesA.size() * movesB.size();
    
    long long nodes = 0;
    for (const auto& moveA : movesA) {
        for (const auto& moveB : movesB) {
            board.make_moves(moveA, moveB);
            nodes += perft(board, depth - 1);
            board.unmake_moves(moveA, moveB);
        }
    }
    return nodes;
}

void benchmark_movegen(int depth) {
    Board board;
    
    // Warmup
    perft(board, 2);
    
    auto start = chrono::high_resolution_clock::now();
    long long nodes = perft(board, depth);
    auto end = chrono::high_resolution_clock::now();
    
    double total_ms = chrono::duration<double, milli>(end - start).count();
    double nps = (nodes * 1000.0) / total_ms;
    
    cout << "=== Perft Benchmark ===" << endl;
    cout << "Depth: " << depth << endl;
    cout << "Nodes: " << nodes << endl;
    cout << "Time: " << total_ms << " ms" << endl;
    cout << "Nodes per second: " << static_cast<long long>(nps) << endl;
    cout << "=======================" << endl;
}
