/*
 * Hivemind - Bughouse Chess Engine
 * RL Settings for self-play training
 * Based on CrazyAra's RLSettings
 */

#pragma once

#include <cstddef>
#include <string>

/**
 * @brief Settings for reinforcement learning self-play.
 * Adapted from CrazyAra for Bughouse chess.
 */
struct RLSettings {
    // Number of games to generate per session (0 = unlimited until samples reached)
    size_t numberOfGames = 0;
    
    // Search settings
    size_t nodesPerMove = 800;           // MCTS nodes per move
    int moveTimeMs = 100;                // Time per move in milliseconds (default 100ms for faster games)
    
    // Node count randomization factor (e.g., 0.2 = +/- 20% nodes)
    float nodeRandomFactor = 0.2f;
    
    // Clips values in policy below this threshold to 0 to reduce noise
    float lowPolicyClipThreshold = 0.01f;
    
    // Playout Cap Randomization (Wu, 2019)
    float quickSearchProbability = 0.0f;  // Probability of quick search
    size_t quickSearchNodes = 100;        // Nodes for quick search
    
    // Resignation settings
    float resignProbability = 0.9f;       // % of games that can be resigned
    float resignThreshold = -0.95f;       // Q-value threshold for resignation
    
    // Temperature for move selection (higher = more exploration)
    float temperature = 1.0f;
    size_t temperatureDecayMoves = 15;    // Moves before temperature decays to 0
    
    // Dirichlet noise for exploration
    float dirichletAlpha = 0.2f;
    float dirichletEpsilon = 0.25f;
    
    // Opening book / starting position file (EPD format)
    std::string epdFilePath = "";
    
    // Output settings
    std::string outputDirectory = "./selfplay_games/";
    std::string pgnFileName = "games.pgn";
    
    // Number of parallel self-play threads
    size_t numThreads = 1;
    
    // Maximum game length (plies) before declaring draw
    size_t maxGameLength = 2048;
    
    // Whether to reuse search tree between moves
    bool reuseTree = true;
    
    // =========================================================================
    // Search Algorithm Parameters (overridable from search_params.h defaults)
    // =========================================================================
    
    // PUCT parameters
    float cpuctInit = 2.5f;              // CPUCT exploration constant
    float cpuctBase = 19652.0f;          // CPUCT base for dynamic scaling
    
    // First Play Urgency
    float fpuReduction = 0.4f;           // FPU reduction from parent Q
    
    // MCGS settings
    bool enableMCGS = true;              // Enable Monte Carlo Graph Search
    bool enableTranspositions = true;    // Enable transposition table
    
    // Draw contempt
    float drawContempt = 0.12f;          // Penalty for draws (0 = neutral)
    
    // Progressive widening
    float pwCoefficient = 1.0f;          // Progressive widening coefficient
    float pwExponent = 0.3f;             // Progressive widening exponent
    
    // Q-value settings
    float qValueWeight = 1.0f;           // Weight for Q-value in move selection
    float qVetoDelta = 0.4f;             // Q-value veto threshold
    
    // =========================================================================
    // Asymmetric Self-Play Settings
    // =========================================================================
    // The team with time advantage is "Alice" (goal: win fast)
    // The team without time advantage is "Bob" (goal: survive/equalize)
    
    // Node multiplier for the time-advantaged team (Alice/attacker)
    // e.g., 0.8 means Alice gets 80% of base nodes (handicap to force efficient play)
    float attackerNodeMultiplier = 0.8f;
    
    // Node multiplier for the time-disadvantaged team (Bob/defender)
    // e.g., 1.5 means Bob gets 150% of base nodes to think about defense
    float defenderNodeMultiplier = 1.5f;
    
    // Time-to-Mate Penalty: Reward scaling based on game length
    // Winner gets: 1.0 - (ply / maxGameLength) * mateSpeedPenalty
    // e.g., mate in 20 ply with maxGameLength=200 and penalty=0.5: reward = 1.0 - (20/200)*0.5 = 0.95
    float mateSpeedPenalty = 0.5f;
    
    // Survival Bonus for the loser (Bob) based on game length
    // Loser gets: -1.0 + (ply / maxGameLength) * survivalBonus
    // e.g., lasting 100 ply with maxGameLength=200 and bonus=0.8: reward = -1.0 + (100/200)*0.8 = -0.6
    float survivalBonus = 0.4f;
    
    // Minimum reward magnitude for winner (even slow wins shouldn't be too low)
    float minWinReward = 0.7f;
    
    // Maximum reward magnitude for loser (even long survivals are still losses)
    float maxLossReward = -0.2f;
};
