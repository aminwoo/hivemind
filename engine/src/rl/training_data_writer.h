/*
 * Hivemind - Bughouse Chess Engine
 * Training Data Writer for Self-Play
 * Outputs training data compatible with Python training pipeline
 * 
 * Uses AlphaZero-style policy targets: normalized MCTS visit distributions
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../constants.h"

/**
 * @brief Sparse policy entry: (move_index, probability)
 * 
 * Storing sparse entries is more efficient since most moves have 0 visits.
 */
struct PolicyEntry {
    uint16_t index;      // Index into UCI_MOVES (max 4672, fits in uint16)
    float probability;   // Normalized visit count
    
    PolicyEntry() : index(0), probability(0.0f) {}
    PolicyEntry(uint16_t idx, float prob) : index(idx), probability(prob) {}
};

/**
 * @brief Represents a single training sample for the neural network.
 * 
 * AlphaZero-style format:
 * - x: Input planes as bytes (64 channels * 8 * 8 = 4096 bytes)
 * - policy_a: Sparse probability distribution over moves for board A (normalized MCTS visits)
 * - policy_b: Sparse probability distribution over moves for board B (normalized MCTS visits)
 * - y_value: Game outcome from perspective of side to move (-1.0, 0.0, or 1.0)
 */
struct TrainingSample {
    std::vector<uint8_t> planes;              // Board planes as uint8 (0 or 1)
    std::vector<PolicyEntry> policyA;         // Sparse policy distribution for board A
    std::vector<PolicyEntry> policyB;         // Sparse policy distribution for board B
    float value;                               // Game outcome from current side's perspective
    
    TrainingSample() : value(0.0f) {}
    
    /**
     * @brief Construct from raw components.
     * @param inputPlanes Float array of input planes
     * @param polA Sparse policy for board A
     * @param polB Sparse policy for board B
     * @param v Value target
     */
    TrainingSample(const float* inputPlanes,
                   const std::vector<PolicyEntry>& polA,
                   const std::vector<PolicyEntry>& polB,
                   float v)
        : policyA(polA), policyB(polB), value(v) {
        // Convert float planes to uint8
        // - Binary planes (pieces, castling, etc.): 0 or 1
        // - Pocket planes (channels 12-21, 44-53): 0-16 (multiply normalized 0.0-1.0 by 16)
        planes.resize(NB_INPUT_VALUES());
        for (size_t i = 0; i < NB_INPUT_VALUES(); ++i) {
            // Check if this is a pocket plane channel
            size_t channel = i / 64;  // 64 squares per channel
            bool isPocketPlane = (channel >= 12 && channel <= 21) || (channel >= 44 && channel <= 53);
            
            if (isPocketPlane) {
                // Store as 0-16: multiply normalized value by 16
                planes[i] = static_cast<uint8_t>(inputPlanes[i] * 16.0f);
            } else {
                // Binary threshold for other planes
                planes[i] = static_cast<uint8_t>(inputPlanes[i] > 0.5f ? 1 : 0);
            }
        }
    }
};

/**
 * @brief Writes training samples to binary files in shards.
 * 
 * Binary format for Python conversion to parquet:
 * - Writes shards of ~65536 samples each
 * - Schema: {x: bytes, policy_a: sparse, policy_b: sparse, y_value: float}
 * 
 * A Python script converts these to parquet with zstd compression.
 */
class TrainingDataWriter {
public:
    /**
     * @brief Construct a new Training Data Writer.
     * @param outputDir Directory to write training data
     * @param samplesPerShard Number of samples per shard file
     */
    TrainingDataWriter(const std::string& outputDir, size_t samplesPerShard = 8192);
    
    ~TrainingDataWriter();
    
    /**
     * @brief Add a training sample to the buffer.
     * Thread-safe: can be called from multiple threads.
     * @param sample The training sample to add
     */
    void add_sample(const TrainingSample& sample);
    
    /**
     * @brief Add a training sample with separate components.
     * Thread-safe: can be called from multiple threads.
     * @param inputPlanes Float array of input planes
     * @param policyA Sparse policy distribution for board A
     * @param policyB Sparse policy distribution for board B
     * @param value Game outcome value
     */
    void add_sample(const float* inputPlanes,
                    const std::vector<PolicyEntry>& policyA,
                    const std::vector<PolicyEntry>& policyB,
                    float value);
    
    /**
     * @brief Flush any remaining samples to disk.
     */
    void flush();
    
    /**
     * @brief Get total number of samples written.
     */
    size_t get_samples_written() const { return samplesWritten.load(); }
    
    /**
     * @brief Get number of shards written.
     */
    size_t get_shards_written() const { return shardsWritten.load(); }

private:
    std::string outputDir;
    size_t samplesPerShard;
    
    std::mutex bufferMutex;
    std::vector<TrainingSample> buffer;
    
    std::atomic<size_t> samplesWritten{0};
    std::atomic<size_t> shardsWritten{0};
    
    /**
     * @brief Write current buffer to a shard file.
     * Called when buffer reaches samplesPerShard size.
     */
    void write_shard();
    
    /**
     * @brief Generate a unique shard filename.
     */
    std::string generate_shard_filename() const;
};

/**
 * @brief Stores samples for a single game, then writes them all at once
 *        when the game result is known.
 * 
 * This is needed because we don't know the game outcome (value) until
 * the game ends.
 */
class GameSampleBuffer {
public:
    /**
     * @brief Add a position sample (planes + policy distributions) without value.
     * Value will be filled in when game ends.
     * @param inputPlanes Float array of input planes
     * @param policyA Sparse policy distribution for board A (normalized MCTS visits)
     * @param policyB Sparse policy distribution for board B (normalized MCTS visits)
     * @param isWhiteTeam Which team made this move (for value sign)
     */
    void add_position(const float* inputPlanes,
                     const std::vector<PolicyEntry>& policyA,
                     const std::vector<PolicyEntry>& policyB,
                     bool isWhiteTeam);
    
    /**
     * @brief Finalize the game with asymmetric rewards and write to the data writer.
     * 
     * Supports different reward values for white and black teams to enable
     * asymmetric training (time-to-mate penalty, survival bonus, etc.)
     * 
     * @param whiteTeamValue Reward for white team positions
     * @param blackTeamValue Reward for black team positions
     * @param writer The training data writer to output samples to
     */
    void finalize_game(float whiteTeamValue, float blackTeamValue, TrainingDataWriter& writer);
    
    /**
     * @brief Clear the buffer without writing (e.g., for abandoned games).
     */
    void clear();
    
    /**
     * @brief Get number of positions stored.
     */
    size_t size() const { return positions.size(); }

private:
    struct PositionData {
        std::vector<uint8_t> planes;
        std::vector<PolicyEntry> policyA;
        std::vector<PolicyEntry> policyB;
        bool isWhiteTeam;  // Used to determine value sign
    };
    
    std::vector<PositionData> positions;
};

/**
 * @brief Helper to build normalized policy distribution from MCTS visit counts.
 * 
 * Takes raw visit counts for moves and converts to a sparse probability distribution.
 * 
 * @param moves Vector of (move_index, visit_count) pairs
 * @param temperature Temperature for visit count exponentiation (1.0 = proportional to visits)
 * @return Sparse policy distribution (normalized to sum to 1.0)
 */
std::vector<PolicyEntry> build_policy_distribution(
    const std::vector<std::pair<uint16_t, int>>& moves,
    float temperature = 1.0f);
