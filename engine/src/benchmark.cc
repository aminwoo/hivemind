#include "benchmark.h"

#include <chrono>
#include <cstdlib>
#include <iostream>

#include "board.h"
#include "constants.h"

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
