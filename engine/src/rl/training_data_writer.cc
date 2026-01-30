/*
 * Hivemind - Bughouse Chess Engine
 * Training Data Writer Implementation
 * 
 * AlphaZero-style training data with normalized MCTS visit distributions
 */

#include "training_data_writer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

using namespace std;
namespace fs = std::filesystem;

// Generate a random hex string for shard IDs
static string generate_random_hex(size_t length) {
    static thread_local mt19937 rng(random_device{}());
    static const char hexChars[] = "0123456789abcdef";
    
    string result;
    result.reserve(length);
    
    uniform_int_distribution<int> dist(0, 15);
    for (size_t i = 0; i < length; ++i) {
        result += hexChars[dist(rng)];
    }
    
    return result;
}

// ============================================================================
// Helper Functions
// ============================================================================

std::vector<PolicyEntry> build_policy_distribution(
    const std::vector<std::pair<uint16_t, int>>& moves,
    float temperature) {
    
    std::vector<PolicyEntry> policy;
    
    if (moves.empty()) {
        return policy;
    }
    
    // Calculate total visits (with temperature applied)
    double totalWeight = 0.0;
    std::vector<double> weights(moves.size());
    
    if (temperature < 0.01f) {
        // Temperature ~0: put all mass on max visit move
        int maxVisits = 0;
        size_t maxIdx = 0;
        for (size_t i = 0; i < moves.size(); ++i) {
            if (moves[i].second > maxVisits) {
                maxVisits = moves[i].second;
                maxIdx = i;
            }
        }
        policy.emplace_back(moves[maxIdx].first, 1.0f);
        return policy;
    }
    
    // Apply temperature: weight = visits^(1/temperature)
    double invTemp = 1.0 / temperature;
    for (size_t i = 0; i < moves.size(); ++i) {
        weights[i] = std::pow(static_cast<double>(moves[i].second), invTemp);
        totalWeight += weights[i];
    }
    
    // Normalize to probabilities
    if (totalWeight > 0.0) {
        policy.reserve(moves.size());
        for (size_t i = 0; i < moves.size(); ++i) {
            float prob = static_cast<float>(weights[i] / totalWeight);
            if (prob > 1e-6f) {  // Skip near-zero probabilities
                policy.emplace_back(moves[i].first, prob);
            }
        }
    }
    
    return policy;
}

// ============================================================================
// TrainingDataWriter Implementation
// ============================================================================

TrainingDataWriter::TrainingDataWriter(const string& outputDir, size_t samplesPerShard)
    : outputDir(outputDir), samplesPerShard(samplesPerShard) {
    // Create output directory if it doesn't exist
    fs::create_directories(outputDir);
    
    // Reserve buffer capacity
    buffer.reserve(samplesPerShard);
}

TrainingDataWriter::~TrainingDataWriter() {
    flush();
}

void TrainingDataWriter::add_sample(const TrainingSample& sample) {
    lock_guard<mutex> lock(bufferMutex);
    buffer.push_back(sample);
    
    if (buffer.size() >= samplesPerShard) {
        write_shard();
    }
}

void TrainingDataWriter::add_sample(const float* inputPlanes,
                                    const std::vector<PolicyEntry>& policyA,
                                    const std::vector<PolicyEntry>& policyB,
                                    float value) {
    TrainingSample sample(inputPlanes, policyA, policyB, value);
    add_sample(sample);
}

void TrainingDataWriter::flush() {
    lock_guard<mutex> lock(bufferMutex);
    if (!buffer.empty()) {
        write_shard();
    }
}

string TrainingDataWriter::generate_shard_filename() const {
    return "shard_" + generate_random_hex(8) + ".bin";
}

/**
 * Binary format v2 for AlphaZero-style policy distributions:
 * 
 * Header (16 bytes):
 *   - Magic bytes: "HVM2" (4 bytes) - v2 for policy distributions
 *   - Version: uint32 (4 bytes) = 2
 *   - Number of samples: uint64 (8 bytes)
 * 
 * Per sample:
 *   - Planes: NB_INPUT_VALUES bytes (4096 for 64 channels * 8 * 8)
 *   - Policy A num entries: uint16 (2 bytes)
 *   - Policy A entries: [uint16 index, float32 prob] * num_entries
 *   - Policy B num entries: uint16 (2 bytes)
 *   - Policy B entries: [uint16 index, float32 prob] * num_entries
 *   - Value: float32 (4 bytes)
 */
void TrainingDataWriter::write_shard() {
    if (buffer.empty()) return;
    
    string filename = generate_shard_filename();
    string filepath = outputDir + "/" + filename;
    
    ofstream file(filepath, ios::binary);
    if (!file) {
        cerr << "Error: Could not open " << filepath << " for writing" << endl;
        return;
    }
    
    // Write header
    const char magic[] = "HVM2";
    file.write(magic, 4);
    
    uint32_t version = 2;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    uint64_t numSamples = buffer.size();
    file.write(reinterpret_cast<const char*>(&numSamples), sizeof(numSamples));
    
    // Write samples
    for (const auto& sample : buffer) {
        // Write planes
        file.write(reinterpret_cast<const char*>(sample.planes.data()), 
                   sample.planes.size());
        
        // Write policy A (sparse)
        uint16_t numEntriesA = static_cast<uint16_t>(sample.policyA.size());
        file.write(reinterpret_cast<const char*>(&numEntriesA), sizeof(numEntriesA));
        for (const auto& entry : sample.policyA) {
            file.write(reinterpret_cast<const char*>(&entry.index), sizeof(entry.index));
            file.write(reinterpret_cast<const char*>(&entry.probability), sizeof(entry.probability));
        }
        
        // Write policy B (sparse)
        uint16_t numEntriesB = static_cast<uint16_t>(sample.policyB.size());
        file.write(reinterpret_cast<const char*>(&numEntriesB), sizeof(numEntriesB));
        for (const auto& entry : sample.policyB) {
            file.write(reinterpret_cast<const char*>(&entry.index), sizeof(entry.index));
            file.write(reinterpret_cast<const char*>(&entry.probability), sizeof(entry.probability));
        }
        
        // Write value
        float value = sample.value;
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    
    file.close();
    
    samplesWritten += buffer.size();
    shardsWritten++;
    
    cout << "Wrote " << buffer.size() << " samples to " << filename 
         << " (total: " << samplesWritten.load() << " samples, " 
         << shardsWritten.load() << " shards)" << endl;
    
    buffer.clear();
    buffer.reserve(samplesPerShard);
}

// ============================================================================
// GameSampleBuffer Implementation
// ============================================================================

void GameSampleBuffer::add_position(const float* inputPlanes,
                                    const std::vector<PolicyEntry>& policyA,
                                    const std::vector<PolicyEntry>& policyB,
                                    bool isWhiteTeam) {
    PositionData pos;
    
    // Convert float planes to uint8
    // - Binary planes (pieces, castling, etc.): 0 or 1
    // - Pocket planes (channels 12-21, 44-53): actual counts 0-16
    pos.planes.resize(NB_INPUT_VALUES());
    for (size_t i = 0; i < NB_INPUT_VALUES(); ++i) {
        // Check if this is a pocket plane channel
        size_t channel = i / 64;  // 64 squares per channel
        bool isPocketPlane = (channel >= 12 && channel <= 21) || (channel >= 44 && channel <= 53);
        
        if (isPocketPlane) {
            // Store actual count (0-16)
            pos.planes[i] = static_cast<uint8_t>(inputPlanes[i] * 16.0f);
        } else {
            // Binary threshold for other planes
            pos.planes[i] = static_cast<uint8_t>(inputPlanes[i] > 0.5f ? 1 : 0);
        }
    }
    
    pos.policyA = policyA;
    pos.policyB = policyB;
    pos.isWhiteTeam = isWhiteTeam;
    
    positions.push_back(std::move(pos));
}

void GameSampleBuffer::finalize_game(float whiteTeamValue, float blackTeamValue, TrainingDataWriter& writer) {
    for (const auto& pos : positions) {
        // Use the appropriate value based on which team made the move
        float value = pos.isWhiteTeam ? whiteTeamValue : blackTeamValue;
        
        TrainingSample sample;
        sample.planes = pos.planes;
        sample.policyA = pos.policyA;
        sample.policyB = pos.policyB;
        sample.value = value;
        
        writer.add_sample(sample);
    }
    
    clear();
}

void GameSampleBuffer::clear() {
    positions.clear();
}
