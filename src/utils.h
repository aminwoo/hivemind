#ifndef UTILS_H
#define UTILS_H

#include <limits>
#include <vector>
#include <cmath>
#include <random>
#include "Fairy-Stockfish/src/types.h"
#include "board.h"
#include "constants.h"

static std::random_device r;
static std::default_random_engine generator(r());

template <typename T>
std::vector<size_t> argsort(const std::vector<T>& v) {
    std::vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

    return idx;
}

template <typename T>
int argmax(T* array, int size) {
    if (size <= 0) {
        return -1; // Handle invalid size
    }

    int max_index = 0; // Start with the first index
    T max_value = array[0]; // Start with the first value

    // Iterate through the array
    for (int i = 1; i < size; i++) {
        if (array[i] > max_value) {
            max_value = array[i]; // Update the maximum value
            max_index = i; // Update the index of the maximum value
        }
    }

    return max_index; // Return the index of the maximum value
}


inline std::string mirror_move(std::string& uciMove) {
    std::string moveMirrored = std::string(uciMove);

    for (unsigned int idx = 1; idx < uciMove.length(); ++idx) {
        if (isdigit(uciMove[idx])) {
            int rank = uciMove[idx] - '0';
            int rank_mirrored = 8 - rank + 1;
            moveMirrored[idx] = char(rank_mirrored + '0');
        }
    }
    return moveMirrored;
}

inline std::vector<float> get_normalized_probablity(float* policyOutput, std::vector<std::pair<int, Stockfish::Move>> actions, Board& board) {
    size_t length = actions.size(); 
    std::vector<float> probs(length);
    
    for(size_t i = 0; i < length; i++) {
        std::pair<int, Stockfish::Move> action = actions[i]; 
        std::string uci = board.uci_move(action.first, action.second);

        if (uci.back() == 'r' || uci.back() == 'b') {
            probs[i] = -std::numeric_limits<float>::infinity(); 
        }
        if (uci.back() == 'q') {
            uci.pop_back();
        }

        if (board.side_to_move(action.first) == Stockfish::BLACK) {
            probs[i] = policyOutput[POLICY_INDEX[mirror_move(uci)]]; 
        }
        else {
            probs[i] = policyOutput[POLICY_INDEX[uci]];
        }
    }

    double sum = 0.0;

    // Compute the exponentials and their sum
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] = exp(probs[i]);
        sum += probs[i];
    }

    // Normalize by dividing by the sum of exponentials
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum;
    }

    return probs;
}

inline std::vector<float> dynamic_vector(float* p, size_t length) {
    std::vector<float> v(length);
    for(size_t i = 0; i < length; i++) {
        v[i] = p[i]; 
    }
    return v; 
}

inline Stockfish::Bitboard flip_vertical(Stockfish::Bitboard x) {
    return __builtin_bswap64(x);
}

inline Stockfish::Square flip_vertical(Stockfish::Square sq) {
    return Stockfish::Square(int(sq) ^ 56);
}

#endif
