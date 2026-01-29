/*
 * Hivemind - Bughouse Chess Engine
 * Model Evaluation - Compare two neural network models
 * Plays tournament games with no exploration noise
 */

#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rl_settings.h"
#include "gamepgn.h"
#include "../board.h"
#include "../agent.h"
#include "../engine.h"

/**
 * @brief Per-player configuration for model evaluation.
 * Allows different search parameters for each side.
 */
struct PlayerConfig {
    // Identity
    std::string name = "Player";
    std::string modelPath = "";     // Path to ONNX model (empty = use default)
    
    // Core search parameters
    size_t nodesPerMove = 800;       // MCTS nodes per move (0 = use time limit instead)
    int moveTimeMs = 0;              // Time per move in milliseconds (0 = use nodes instead)
    int batchSize = 8;               // Batch size for neural network inference
    int numSearchThreads = 2;        // Number of search threads
    
    // PUCT parameters
    float cpuctInit = 2.5f;          // PUCT exploration constant
    float cpuctBase = 19652.0f;      // PUCT base for dynamic scaling
    
    // First Play Urgency
    float fpuReduction = 0.4f;       // FPU reduction from parent Q
    
    // Dirichlet noise (for exploration during eval if desired)
    float dirichletEpsilon = 0.0f;   // Dirichlet noise epsilon (0 = no noise for eval)
    float dirichletAlpha = 0.2f;     // Dirichlet noise alpha
    
    // Temperature (applies to first N moves, then decays to 0)
    float temperature = 0.5f;        // Move selection temperature (0 = greedy, 1.0 = full stochastic)
    size_t temperatureDecayMoves = 30; // Number of moves before temperature decays to 0
    
    // MCGS (Monte Carlo Graph Search) settings
    bool enableMCGS = true;          // Enable MCGS with transposition table
    bool enableTranspositions = true; // Enable transposition table lookups
    
    // Draw contempt
    float drawContempt = 0.15f;      // Penalty for draws (0 = neutral)
    
    // Progressive widening
    float pwCoefficient = 1.0f;      // Progressive widening coefficient
    float pwExponent = 0.3f;         // Progressive widening exponent
    
    // Q-value settings
    float qValueWeight = 1.0f;       // Weight for Q-value in move selection
    float qVetoDelta = 0.4f;         // Q-value veto threshold
    
    // Convenience methods
    bool hasCustomModel() const { return !modelPath.empty(); }
};

/**
 * @brief Statistics for a model evaluation tournament.
 */
struct EvalStats {
    // Win/Draw/Loss from Player 1's perspective
    size_t player1Wins = 0;
    size_t player1Losses = 0;
    size_t draws = 0;
    
    // Game length statistics
    size_t totalPlies = 0;
    size_t minGameLength = SIZE_MAX;
    size_t maxGameLength = 0;
    
    // Opening move tracking (first N joint moves)
    std::map<std::string, size_t> openingMoves;  // "e4,e4" -> count
    
    // Per-color stats (to detect first-move advantage)
    size_t player1WinsAsWhite = 0;
    size_t player1WinsAsBlack = 0;
    size_t player1LossesAsWhite = 0;
    size_t player1LossesAsBlack = 0;
    
    float winRate() const {
        size_t total = player1Wins + player1Losses + draws;
        return total > 0 ? (100.0f * player1Wins / total) : 0.0f;
    }
    
    float drawRate() const {
        size_t total = player1Wins + player1Losses + draws;
        return total > 0 ? (100.0f * draws / total) : 0.0f;
    }
    
    float lossRate() const {
        size_t total = player1Wins + player1Losses + draws;
        return total > 0 ? (100.0f * player1Losses / total) : 0.0f;
    }
    
    float avgGameLength() const {
        size_t total = player1Wins + player1Losses + draws;
        return total > 0 ? (static_cast<float>(totalPlies) / total) : 0.0f;
    }
    
    /**
     * @brief Calculate Elo difference based on win rate.
     * Uses the formula: Elo = -400 * log10(1/score - 1)
     * where score = (wins + 0.5*draws) / total
     */
    float eloDifference() const {
        size_t total = player1Wins + player1Losses + draws;
        if (total == 0) return 0.0f;
        
        float score = (player1Wins + 0.5f * draws) / total;
        if (score <= 0.0f) return -999.0f;
        if (score >= 1.0f) return 999.0f;
        
        return -400.0f * std::log10(1.0f / score - 1.0f);
    }
};

/**
 * @brief Model evaluation settings.
 */
struct EvalSettings {
    size_t numGames = 100;           // Number of games to play
    size_t nodesPerMove = 800;       // MCTS nodes per move (default, can be overridden per player)
    float temperature = 0.6f;        // Temperature for move selection (0.0 = deterministic, higher = more random)
    size_t maxGameLength = 2048;     // Maximum plies before draw
    size_t openingMovesToTrack = 4;  // Number of opening moves to track
    bool verbose = false;            // Print each game result
    std::string outputPgnPath = ""; // Optional: save games to PGN file
    
    // Per-player configurations for parameter testing
    PlayerConfig player1;            // First player ("new" model in classic eval)
    PlayerConfig player2;            // Second player ("old" model in classic eval)
    bool usePlayerConfigs = false;   // If true, use player1/player2 instead of nodesPerMove
};

/**
 * @brief Model evaluation manager for comparing two models.
 * 
 * Plays a tournament between two models with deterministic play
 * (no exploration noise, temperature = 0) to measure relative strength.
 * 
 * Games alternate which model plays as white team to avoid bias.
 */
class ModelEvaluator {
public:
    /**
     * @brief Construct a new ModelEvaluator.
     * @param newModelEngines Engines loaded with the new (challenger) model
     * @param oldModelEngines Engines loaded with the old (baseline) model
     * @param settings Evaluation settings
     */
    ModelEvaluator(const std::vector<Engine*>& newModelEngines,
                   const std::vector<Engine*>& oldModelEngines,
                   const EvalSettings& settings);
    
    ~ModelEvaluator();
    
    // Prevent copying
    ModelEvaluator(const ModelEvaluator&) = delete;
    ModelEvaluator& operator=(const ModelEvaluator&) = delete;
    
    /**
     * @brief Run the evaluation tournament.
     * @return EvalStats with tournament results
     */
    EvalStats run();
    
    /**
     * @brief Stop the evaluation early.
     */
    void stop();
    
    /**
     * @brief Check if evaluation is currently running.
     */
    bool isRunning() const { return running; }
    
    /**
     * @brief Get current statistics (can be called while running).
     */
    EvalStats getStats() const;
    
    /**
     * @brief Print formatted statistics to stdout.
     */
    void printStats() const;

private:
    const std::vector<Engine*>& newModelEngines;
    const std::vector<Engine*>& oldModelEngines;
    const EvalSettings& settings;
    
    // Agents for each model
    std::unique_ptr<Agent> newModelAgent;
    std::unique_ptr<Agent> oldModelAgent;
    
    // Runtime state
    std::atomic<bool> running;
    std::chrono::steady_clock::time_point startTime;
    
    // Statistics (protected by mutex for thread-safe access)
    mutable std::mutex statsMutex;
    EvalStats stats;
    
    // PGN output
    std::mutex pgnMutex;
    
    /**
     * @brief Play a single evaluation game.
     * @param newModelIsWhite If true, new model plays as white team
     * @param gameNumber Game number for logging
     * @return Game result from white team's perspective
     */
    GameResult playGame(bool newModelIsWhite, size_t gameNumber);
    
    /**
     * @brief Record the opening moves from a game.
     * @param moves Vector of (boardNum, moveStr) pairs
     */
    void recordOpeningMoves(const std::vector<std::pair<int, std::string>>& moves);
    
    /**
     * @brief Write a game to the PGN file.
     */
    void writeGameToPgn(const BughouseGamePGN& pgn);
    
    /**
     * @brief Get top N most common opening sequences.
     */
    std::vector<std::pair<std::string, size_t>> getTopOpenings(size_t n) const;
};

/**
 * @brief Entry point for model evaluation.
 * @param newModelPath Path to new model ONNX file
 * @param oldModelPath Path to old model ONNX file  
 * @param settings Evaluation settings
 */
void run_model_eval(const std::string& newModelPath,
                    const std::string& oldModelPath,
                    const EvalSettings& settings);

/**
 * @brief Entry point for parameter evaluation (same model, different settings).
 * Tests the same model against itself with different search parameters
 * to evaluate the impact of parameters like batch size, CPUCT, etc.
 * 
 * @param modelPath Path to model ONNX file (used for both players)
 * @param settings Evaluation settings with player1/player2 configurations
 */
void run_param_eval(const std::string& modelPath,
                    const EvalSettings& settings);
