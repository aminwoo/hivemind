#pragma once
#include <limits>
#include <vector>
#include <cmath>
#include <numeric>   
#include <algorithm>  
#include <type_traits>
#include <cuda_fp16.h>
#include "Fairy-Stockfish/src/types.h"
#include "environment/board.h"
#include "environment/constants.h"
#include "environment/joint_action.h"
#include "common/globals.h"

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

inline std::vector<float> normalize_logits(const std::vector<float>& logits) {
    std::vector<float> probabilities(logits.size(), 0.0f);
    float maxLogit = -std::numeric_limits<float>::infinity();
    for (float logit : logits) {
        if (std::isfinite(logit)) {
            maxLogit = std::max(maxLogit, logit);
        }
    }

    if (!std::isfinite(maxLogit)) {
        if (!probabilities.empty()) {
            std::fill(probabilities.begin(), probabilities.end(), 1.0f / probabilities.size());
        }
        return probabilities;
    }

    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (std::isfinite(logits[i])) {
            probabilities[i] = std::exp(logits[i] - maxLogit);
            sum += probabilities[i];
        }
    }

    if (!std::isfinite(sum) || sum <= 0.0) {
        size_t finiteCount = std::count_if(logits.begin(), logits.end(), [](float value) {
            return std::isfinite(value);
        });
        if (finiteCount > 0) {
            for (size_t i = 0; i < logits.size(); ++i) {
                probabilities[i] = std::isfinite(logits[i]) ? 1.0f / finiteCount : 0.0f;
            }
        }
        return probabilities;
    }

    for (float& probability : probabilities) {
        probability = static_cast<float>(probability / sum);
    }
    return probabilities;
}

inline bool is_policy_move_representable(Board& board,
                                         int boardNum,
                                         Stockfish::Move move) {
    if (move == Stockfish::MOVE_NONE) {
        return true;
    }
    if (Stockfish::type_of(move) == Stockfish::PROMOTION) {
        Stockfish::PieceType pt = Stockfish::promotion_type(move);
        if (pt == Stockfish::ROOK || pt == Stockfish::BISHOP) {
            return false;
        }
    }
    return true;
}

inline int get_fast_policy_index(Stockfish::Move move, Stockfish::Color sideToMove) {
    if (move == Stockfish::MOVE_NONE) {
        return 0; // pass
    }
    const int color = (sideToMove == Stockfish::BLACK) ? 1 : 0;
    if (Stockfish::type_of(move) == Stockfish::DROP) {
        const int to_sq = static_cast<int>(Stockfish::to_sq(move));
        const int pt = static_cast<int>(Stockfish::dropped_piece_type(move));
        if (to_sq >= 0 && to_sq < 64 && pt >= 1 && pt <= 5) {
            return POLICY_TABLE_DROP[color][to_sq][pt];
        }
        return -1;
    }

    const int from_sq = static_cast<int>(Stockfish::from_sq(move));
    const int to_sq = static_cast<int>(Stockfish::to_sq(move));
    if (from_sq < 0 || from_sq >= 64 || to_sq < 0 || to_sq >= 64) {
        return -1;
    }

    if (Stockfish::type_of(move) == Stockfish::PROMOTION) {
        const Stockfish::PieceType pt = Stockfish::promotion_type(move);
        if (pt == Stockfish::KNIGHT) {
            return POLICY_TABLE_NORMAL[color][from_sq][to_sq][1];
        }
        if (pt == Stockfish::QUEEN) {
            return POLICY_TABLE_NORMAL[color][from_sq][to_sq][0];
        }
        return -1; // Underpromotion to R or B not in policy representation
    }

    return POLICY_TABLE_NORMAL[color][from_sq][to_sq][0];
}

template <typename T>
inline float policy_value_to_float(T value) {
    if constexpr (std::is_same_v<T, __half>) {
        return __half2float(value);
    }
    return value;
}

template <typename T>
inline std::vector<float> get_normalized_probability(
    const T* policyOutput, const std::vector<int>& policyIndices) {
    std::vector<float> logits(policyIndices.size());
    for (size_t i = 0; i < policyIndices.size(); ++i) {
        const int policyIdx = policyIndices[i];
        logits[i] = policyIdx >= 0
            ? policy_value_to_float(policyOutput[policyIdx])
            : -std::numeric_limits<float>::infinity();
    }
    return normalize_logits(logits);
}

template <typename T>
inline std::vector<float> get_normalized_probability(
    const T* policyOutput, const std::vector<Stockfish::Move>& actions,
    int board_num, Board& board) {
    const Stockfish::Color stm = board.side_to_move(board_num);
    std::vector<int> policyIndices;
    policyIndices.reserve(actions.size());
    for (Stockfish::Move action : actions) {
        policyIndices.push_back(get_fast_policy_index(action, stm));
    }
    return get_normalized_probability(policyOutput, policyIndices);
}
