/*
 * Hivemind - Bughouse Chess Engine
 * Self-play for reinforcement learning
 * Adapted from CrazyAra's SelfPlay
 */

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rl_settings.h"
#include "gamepgn.h"
#include "training_data_writer.h"
#include "../board.h"
#include "../agent.h"
#include "../engine.h"
#include "../planes.h"
#include "../globals.h"

/**
 * @brief Self-play manager for Bughouse reinforcement learning.
 * 
 * Key difference from standard chess self-play:
 * - One team always has timeAdvantage=true, the other has timeAdvantage=false
 * - This creates asymmetric training data where one team can "sit" more freely
 * - Games are output in Bughouse PGN format with board prefixes (1/2)
 * - Training data includes AlphaZero-style MCTS visit distributions
 */
class SelfPlay {
public:
    /**
     * @brief Construct a new SelfPlay object.
     * @param settings RL settings for self-play
     * @param engines Vector of Engine pointers for neural network inference
     */
    SelfPlay(const RLSettings& settings, const std::vector<Engine*>& engines);
    
    ~SelfPlay();
    
    // Prevent copying
    SelfPlay(const SelfPlay&) = delete;
    SelfPlay& operator=(const SelfPlay&) = delete;
    
    /**
     * @brief Start self-play game generation.
     * @param numberOfGames Number of games to generate (0 = use settings)
     */
    void go(size_t numberOfGames = 0);
    
    /**
     * @brief Stop the self-play generation.
     */
    void stop();
    
    /**
     * @brief Check if self-play is currently running.
     */
    bool is_running() const { return running; }
    
    /**
     * @brief Get the number of games generated so far.
     */
    size_t get_games_generated() const { return gamesGenerated; }
    
    /**
     * @brief Get the total number of samples (positions) generated.
     */
    size_t get_samples_generated() const { return samplesGenerated; }

private:
    const RLSettings& settings;
    const std::vector<Engine*>& engines;
    
    std::unique_ptr<Agent> agentWhiteTeam;  // Agent for white team (has time advantage)
    std::unique_ptr<Agent> agentBlackTeam;  // Agent for black team (no time advantage)
    
    std::atomic<bool> running;
    std::atomic<size_t> gamesGenerated;
    std::atomic<size_t> samplesGenerated;
    
    std::mutex pgnFileMutex;
    std::string pgnFilePath;
    
    // Training data output
    std::unique_ptr<TrainingDataWriter> trainingDataWriter;
    
    // Statistics
    std::chrono::steady_clock::time_point startTime;
    float gamesPerMinute;
    
    // Alice (attacker with time advantage) vs Bob (defender) statistics
    std::atomic<size_t> aliceWins{0};
    std::atomic<size_t> aliceLosses{0};
    std::atomic<size_t> aliceDraws{0};
    
    // Game length tracking
    std::atomic<size_t> totalPlies{0};
    std::atomic<size_t> minGameLength{999999};
    std::atomic<size_t> maxGameLength{0};
    
    /**
     * @brief Generate a single self-play game.
     * @param whiteHasTimeAdvantage If true, white team has time advantage
     * @param verbose Print game progress to stdout
     * @return The game result
     */
    GameResult generate_game(bool whiteHasTimeAdvantage, bool verbose);
    
    /**
     * @brief Extract policy distributions from MCTS root node.
     * Marginalizes joint action visits to per-board distributions.
     * @param rootNode The MCTS root node after search
     * @param board The current board state (for move-to-UCI conversion)
     * @param policyA Output: sparse policy distribution for board A
     * @param policyB Output: sparse policy distribution for board B
     * @return Vector of (JointActionCandidate, visit_count) pairs for temperature sampling
     */
    std::vector<std::pair<JointActionCandidate, int>> extract_policy_distributions(
                                      const std::shared_ptr<Node>& rootNode,
                                      Board& board,
                                      std::vector<PolicyEntry>& policyA,
                                      std::vector<PolicyEntry>& policyB);
    
    /**
     * @brief Write a completed game to the PGN file.
     * @param pgn The game to write
     * @param verbose Also print to stdout
     */
    void write_game_to_pgn(const BughouseGamePGN& pgn, bool verbose);
    
    /**
     * @brief Check if the game should be resigned based on evaluation.
     * @param eval Current evaluation
     * @param allowResignation Whether resignation is allowed for this game
     * @return True if should resign
     */
    bool should_resign(float eval, bool allowResignation) const;
    
    /**
     * @brief Apply temperature to move selection based on ply count.
     * @param ply Current ply number
     * @return Temperature value
     */
    float get_temperature(size_t ply) const;
    
    /**
     * @brief Randomly adjust node count for variety.
     * @param baseNodes Base node count
     * @return Adjusted node count
     */
    size_t randomize_nodes(size_t baseNodes) const;
    
    /**
     * @brief Calculate asymmetric rewards based on game outcome and length.
     * 
     * Implements:
     * - Time-to-Mate Penalty: Winner gets higher reward for faster wins
     * - Survival Bonus: Loser gets less negative reward for lasting longer
     * 
     * @param result Game result
     * @param ply Number of plies in the game
     * @param whiteHadTimeAdvantage Whether white team had time advantage
     * @return Pair of (whiteTeamValue, blackTeamValue)
     */
    std::pair<float, float> calculate_asymmetric_rewards(
        GameResult result, size_t ply, bool whiteHadTimeAdvantage) const;
    
    /**
     * @brief Print speed statistics.
     */
    void print_statistics() const;
};

/**
 * @brief Entry point for selfplay command.
 * @param settings RL settings
 * @param engines Available engines
 * @param numberOfGames Number of games (0 = from settings)
 */
void run_selfplay(const RLSettings& settings, const std::vector<Engine*>& engines, size_t numberOfGames);
