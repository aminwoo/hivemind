#pragma once
#include <limits>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>   
#include <algorithm>  
#include "Fairy-Stockfish/src/types.h"
#include "board.h"
#include "constants.h"
#include "joint_action.h"

// Global random engine for generating random numbers.
static std::random_device r;
static std::default_random_engine generator(r());

/**
 * @brief Returns indices that sort the vector in descending order.
 *
 * Given a vector of values, this function returns a vector of indices
 * such that iterating over the indices yields the original vector's values in descending order.
 *
 * @tparam T Type of the vector elements.
 * @param v Input vector.
 * @return std::vector<size_t> Indices that would sort the vector in descending order.
 */
template <typename T>
std::vector<size_t> argsort(const std::vector<T>& v) {
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0); // Fill indices: 0, 1, ..., n-1

    std::sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; }); // Sort indices by corresponding values

    return idx;
}

/**
 * @brief Finds the index of the maximum element in an array.
 *
 * Scans a raw array and returns the index of the maximum element.
 * Returns -1 if the provided size is not positive.
 *
 * @tparam T Type of the array elements.
 * @param array Pointer to the first element of the array.
 * @param size Number of elements in the array.
 * @return int Index of the maximum element, or -1 if size is invalid.
 */
template <typename T>
int argmax(T* array, int size) {
    if (size <= 0) {
        return -1; // Invalid size
    }

    int max_index = 0;       // Initialize to the first index
    T max_value = array[0];  // Initialize to the first value

    // Iterate to find the maximum value and its index
    for (int i = 1; i < size; i++) {
        if (array[i] > max_value) {
            max_value = array[i];
            max_index = i;
        }
    }

    return max_index;
}

/**
 * @brief Mirrors a UCI move string vertically.
 *
 * Adjusts the rank digits in the move string (e.g., "e2e4") to reflect a vertical flip,
 * effectively mirroring the move on the board.
 *
 * @param uciMove UCI move string.
 * @return std::string Mirrored UCI move string.
 */
inline std::string mirror_move(std::string& uciMove) {
    std::string moveMirrored = uciMove;

    // Start from index 1 to skip the file letter and update rank digits
    for (unsigned int idx = 1; idx < uciMove.length(); ++idx) {
        if (isdigit(uciMove[idx])) {
            int rank = uciMove[idx] - '0';
            int rank_mirrored = 8 - rank + 1;
            moveMirrored[idx] = char(rank_mirrored + '0');
        }
    }
    return moveMirrored;
}

/**
 * @brief Converts a raw array of floats into a std::vector<float>.
 *
 * @param p Pointer to the float array.
 * @param length Number of elements in the array.
 * @return std::vector<float> Vector containing the array's elements.
 */
inline std::vector<float> dynamic_vector(float* p, size_t length) {
    std::vector<float> v(length);
    for (size_t i = 0; i < length; i++) {
        v[i] = p[i];
    }
    return v;
}

/**
 * @brief Flips a 64-bit bitboard vertically.
 *
 * Uses a built-in byte swap to reverse the order of bytes, effectively flipping the bitboard.
 *
 * @param x Bitboard to flip.
 * @return Stockfish::Bitboard Vertically flipped bitboard.
 */
inline Stockfish::Bitboard flip_vertical(Stockfish::Bitboard x) {
    return __builtin_bswap64(x);
}

/**
 * @brief Flips a square index vertically.
 *
 * Applies an XOR with 56 to map each square to its vertically mirrored counterpart.
 *
 * @param sq Square index to flip.
 * @return Stockfish::Square Vertically flipped square index.
 */
inline Stockfish::Square flip_vertical(Stockfish::Square sq) {
    return Stockfish::Square(int(sq) ^ 56);
}

inline std::vector<float> get_normalized_probability(float* policyOutput,
std::vector<Stockfish::Move> actions,
int board_num, Board& board, bool prioritizePass = false) {
    size_t length = actions.size();
    std::vector<float> logits(length);
    int passIdx = -1;
    float maxLogit = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < length; i++) {
        Stockfish::Move action = actions[i];
        std::string uci = board.uci_move(board_num, action);
        
        // Track which index is the pass move
        if (uci == "pass") {
            passIdx = static_cast<int>(i);
        }
        
        // Treat queen underpromotion as default move
        if (uci.back() == 'q') {
            uci.pop_back();
        }
        
        // Mirror move for Black's perspective
        if (board.side_to_move(board_num) == Stockfish::BLACK) {
            logits[i] = policyOutput[POLICY_INDEX[mirror_move(uci)]];
        } else {
            logits[i] = policyOutput[POLICY_INDEX[uci]];
        }
        
        // Rook or bishop underpromotions should be ignored
        if (uci.back() == 'r' || uci.back() == 'b') {
            logits[i] = -std::numeric_limits<float>::infinity();
        }
        
        // Track max logit (excluding pass for now)
        if (uci != "pass" && logits[i] > maxLogit) {
            maxLogit = logits[i];
        }
    }
    
    // If prioritizePass is true, boost pass move logit towards the argmax move's logit
    // This ensures pass is a competitive option without completely overriding the network's preference
    // Use a weighted average: boosted_pass = pass_logit * (1 - alpha) + max_logit * alpha
    if (prioritizePass && passIdx >= 0 && maxLogit > -std::numeric_limits<float>::infinity()) {
        constexpr float PASS_BOOST_ALPHA = 0.85f;  // 0 = no boost, 1 = full boost to argmax
        float passLogit = logits[passIdx];
        logits[passIdx] = passLogit * (1.0f - PASS_BOOST_ALPHA) + maxLogit * PASS_BOOST_ALPHA;
    }
    
    std::vector<float> probs(length);
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = exp(logits[i]);
        sum += probs[i];
    }
    
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum;
    }
    return probs;
}

// ============================================================================
// Random Sampling Utilities
// ============================================================================

// Thread-local random generator for sampling operations
inline std::mt19937& get_thread_local_rng() {
    static thread_local std::mt19937 rng(std::random_device{}());
    return rng;
}

/**
 * @brief Generate Dirichlet noise vector.
 * @param length Number of elements
 * @param alpha Concentration parameter (smaller = more noise, larger = more uniform)
 * @return Normalized Dirichlet sample (sums to 1.0)
 */
inline std::vector<float> generate_dirichlet_noise(size_t length, float alpha) {
    std::vector<float> noise(length);
    float sum = 0.0f;
    
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    auto& rng = get_thread_local_rng();
    for (size_t i = 0; i < length; ++i) {
        noise[i] = gamma(rng);
        sum += noise[i];
    }
    
    // Normalize to sum to 1
    if (sum > 0.0f) {
        for (size_t i = 0; i < length; ++i) {
            noise[i] /= sum;
        }
    }
    
    return noise;
}

/**
 * @brief Sample an index based on visit counts with temperature.
 * 
 * Uses the formula: P(i) = visits[i]^(1/T) / sum(visits^(1/T))
 * 
 * @param visits Vector of visit counts
 * @param temperature Temperature for sampling (0 = greedy, >0 = stochastic)
 * @return Sampled index
 */
inline size_t sample_index_with_temperature(const std::vector<int>& visits, float temperature) {
    if (visits.empty()) return 0;
    
    // Greedy selection for temperature near 0
    if (temperature < 0.01f) {
        size_t bestIdx = 0;
        int maxVisits = visits[0];
        for (size_t i = 1; i < visits.size(); ++i) {
            if (visits[i] > maxVisits) {
                maxVisits = visits[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }
    
    // Temperature-based sampling: P(i) = visits[i]^(1/T) / sum(visits^(1/T))
    std::vector<double> probs(visits.size());
    double sum = 0.0;
    double invTemp = 1.0 / temperature;
    
    for (size_t i = 0; i < visits.size(); ++i) {
        probs[i] = std::pow(static_cast<double>(visits[i]), invTemp);
        sum += probs[i];
    }
    
    if (sum <= 0.0) {
        return 0;
    }
    
    // Normalize
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum;
    }
    
    // Sample
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(get_thread_local_rng());
    double cumulative = 0.0;
    
    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative += probs[i];
        if (r <= cumulative) {
            return i;
        }
    }
    
    return visits.size() - 1;
}

/**
 * @brief Sample a joint action from visit distribution with temperature.
 * Uses the sample_index_with_temperature utility function.
 * @param childActionVisits Vector of (JointActionCandidate, visit_count) pairs.
 * @param temperature Temperature for sampling (0 = greedy, >0 = stochastic).
 * @return Sampled joint action.
 */
inline JointActionCandidate sample_action_with_temperature(
    const std::vector<std::pair<JointActionCandidate, int>>& childActionVisits,
    float temperature
) {
    if (childActionVisits.empty()) {
        return JointActionCandidate();
    }
    
    // Extract visit counts into a vector for the utility function
    std::vector<int> visits(childActionVisits.size());
    for (size_t i = 0; i < childActionVisits.size(); ++i) {
        visits[i] = childActionVisits[i].second;
    }
    
    // Use the shared utility function
    size_t selectedIdx = sample_index_with_temperature(visits, temperature);
    return childActionVisits[selectedIdx].first;
}
