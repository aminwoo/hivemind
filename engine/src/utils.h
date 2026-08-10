#pragma once
#include <limits>
#include <vector>
#include <cmath>
#include <numeric>   
#include <algorithm>  
#include "Fairy-Stockfish/src/types.h"
#include "board.h"
#include "constants.h"
#include "joint_action.h"

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

inline void apply_probability_floor(std::vector<float>& probabilities,
                                    size_t targetIdx,
                                    float probabilityFloor) {
    if (targetIdx >= probabilities.size()) {
        return;
    }

    probabilityFloor = std::clamp(probabilityFloor, 0.0f, 1.0f);
    const float currentProbability = probabilities[targetIdx];
    if (currentProbability >= probabilityFloor || currentProbability >= 1.0f) {
        return;
    }

    const float scale = (1.0f - probabilityFloor) / (1.0f - currentProbability);
    for (size_t idx = 0; idx < probabilities.size(); ++idx) {
        if (idx != targetIdx) {
            probabilities[idx] *= scale;
        }
    }
    probabilities[targetIdx] = probabilityFloor;
}

inline float get_pass_prior_floor(bool teamHasTimeAdvantage,
                                  bool boardAOnTurn,
                                  bool boardBOnTurn,
                                  const SearchParams::RuntimeConfig& config) {
    if (is_double_sit_legal(teamHasTimeAdvantage, boardAOnTurn, boardBOnTurn)) {
        return config.waitPassPriorFloor;
    }
    if (boardAOnTurn && boardBOnTurn) {
        return config.coordinationPassPriorFloor;
    }
    return 0.0f;
}

inline bool is_policy_move_representable(Board& board,
                                         int boardNum,
                                         Stockfish::Move move) {
    if (move == Stockfish::MOVE_NONE) {
        return true;
    }
    const std::string uci = board.uci_move(boardNum, move);
    return uci.size() != 5 || (uci.back() != 'r' && uci.back() != 'b');
}

inline std::vector<float> get_normalized_probability(float* policyOutput,
const std::vector<Stockfish::Move>& actions,
int board_num, Board& board, float passPriorFloor = 0.0f) {
    size_t length = actions.size();
    std::vector<float> logits(length);
    size_t passIdx = actions.size();

    for (size_t i = 0; i < length; i++) {
        Stockfish::Move action = actions[i];
        if (action == Stockfish::MOVE_NONE) {
            passIdx = i;
        }
        if (!is_policy_move_representable(board, board_num, action)) {
            logits[i] = -std::numeric_limits<float>::infinity();
            continue;
        }
        std::string uci = board.uci_move(board_num, action);
        
        // Treat queen underpromotion as default move
        if (uci.back() == 'q') {
            uci.pop_back();
        }
        
        // Mirror move for Black's perspective
        std::string policyMove = board.side_to_move(board_num) == Stockfish::BLACK
            ? mirror_move(uci)
            : uci;
        auto policyIt = POLICY_INDEX.find(policyMove);
        logits[i] = policyIt != POLICY_INDEX.end()
            ? policyOutput[policyIt->second]
            : -std::numeric_limits<float>::infinity();
    }

    std::vector<float> probabilities = normalize_logits(logits);
    apply_probability_floor(probabilities, passIdx, passPriorFloor);
    return probabilities;
}
