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
 * @brief Computes normalized probabilities for a set of chess moves.
 *
 * Processes policy network outputs for given actions by adjusting UCI move strings based on board state,
 * applying penalties, and then performing a softmax normalization.
 *
 * @param policyOutput Array of raw policy outputs.
 * @param actions Vector of action pairs (identifier and move).
 * @param board Current board state.
 * @return std::vector<float> Normalized probability distribution over actions.
 */
inline std::vector<float> get_normalized_probablity(float* policyOutput,
                                                    std::vector<std::pair<int, Stockfish::Move>> actions,
                                                    Board& board) {
    size_t length = actions.size(); 
    std::vector<float> probs(length);
    
    // Process each action to extract and adjust the move string
    for (size_t i = 0; i < length; i++) {
        std::pair<int, Stockfish::Move> action = actions[i]; 
        std::string uci = board.uci_move(action.first, action.second);

        // Treat queen underpromotion as default move
        if (uci.back() == 'q') {
            uci.pop_back();
        }

        // Mirror move for Black's perspective
        if (board.side_to_move(action.first) == Stockfish::BLACK) {
            probs[i] = policyOutput[POLICY_INDEX[mirror_move(uci)]];
        } else {
            probs[i] = policyOutput[POLICY_INDEX[uci]];
        }

        // Rook or bishop underpromotions should be ignored 
        if (uci.back() == 'r' || uci.back() == 'b') {
            probs[i] = -std::numeric_limits<float>::infinity();
        }
    }

    double sum = 0.0;

    // Apply exponentiation and compute the sum for softmax normalization
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] = exp(probs[i]);
        sum += probs[i];
    }

    // Normalize probabilities by dividing each by the sum
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum;
    }

    return probs;
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