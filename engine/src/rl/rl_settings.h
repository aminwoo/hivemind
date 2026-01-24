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
    size_t temperatureDecayMoves = 30;    // Moves before temperature decays to 0
    
    // Dirichlet noise for exploration
    float dirichletAlpha = 0.3f;
    float dirichletEpsilon = 0.25f;
    
    // Opening book / starting position file (EPD format)
    std::string epdFilePath = "";
    
    // Output settings
    std::string outputDirectory = "./selfplay_games/";
    std::string pgnFileName = "games.pgn";
    
    // Number of parallel self-play threads
    size_t numThreads = 1;
    
    // Maximum game length (plies) before declaring draw
    size_t maxGameLength = 500;
    
    // Whether to reuse search tree between moves
    bool reuseTree = true;
};
