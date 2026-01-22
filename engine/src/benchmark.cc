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

static long long perft(Board& board, int boardNum, int depth) {
    if (depth == 0) return 1;
    
    auto moves = board.legal_moves(boardNum);
    if (depth == 1) return moves.size();
    
    long long nodes = 0;
    for (const auto& move : moves) {
        if (boardNum == BOARD_A) {
            board.make_moves(move, Stockfish::MOVE_NONE);
        } else {
            board.make_moves(Stockfish::MOVE_NONE, move);
        }
        nodes += perft(board, boardNum, depth - 1);
        if (boardNum == BOARD_A) {
            board.unmake_moves(move, Stockfish::MOVE_NONE);
        } else {
            board.unmake_moves(Stockfish::MOVE_NONE, move);
        }
    }
    return nodes;
}

void benchmark_movegen(int depth) {
    Board board;
    
    // Warmup
    perft(board, BOARD_A, 3);
    
    auto start = chrono::high_resolution_clock::now();
    long long nodes = perft(board, BOARD_A, depth);
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
