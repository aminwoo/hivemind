#ifndef UTILS_H
#define UTILS_H

#include <blaze/Math.h>
#include <random>
#include "Fairy-Stockfish/src/types.h"
#include "bugboard.h"
#include "constants.h"

using blaze::StaticVector;
using blaze::DynamicVector;

static std::random_device r;
static std::default_random_engine generator(r());

template <typename T>
DynamicVector<T> get_dirichlet_noise(size_t length, T alpha) {
    DynamicVector<T> dirichletNoise(length);

    for (size_t i = 0; i < length; ++i) {
        std::gamma_distribution<T> distribution(alpha, 1.0f);
        dirichletNoise[i] = distribution(generator);
    }
    dirichletNoise /= sum(dirichletNoise);
    return  dirichletNoise;
}

template <typename T>
std::vector<size_t> argsort(const DynamicVector<T>& v) {
    std::vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

    return idx;
}

inline std::string mirror_move(std::string& uciMove) {
    std::string moveMirrored = std::string(uciMove);

    for (unsigned int idx = 0; idx < uciMove.length(); ++idx) {
        if (isdigit(uciMove[idx])) {
            int rank = uciMove[idx] - '0';
            int rank_mirrored = 8 - rank + 1;
            moveMirrored[idx] = char(rank_mirrored + '0');
        }
    }
    return moveMirrored;
}

inline DynamicVector<float> get_normalized_probablity(float* policyOutput, std::vector<std::pair<int, Stockfish::Move>> actions, Bugboard& board) {
    size_t length = actions.size(); 
    DynamicVector<float> probs(length);
    
    for(size_t i = 0; i < length; i++) {
        std::pair<int, Stockfish::Move> action = actions[i]; 
        std::string uci = board.uci_move(action.first, action.second);
        if (action.first == 0) {
            if (board.side_to_move(0) == Stockfish::BLACK) {
                probs[i] = policyOutput[POLICY_INDEX[mirror_move(uci)]]; 
            }
            else {
                probs[i] = policyOutput[POLICY_INDEX[uci]];
            }
        }
        else {
            if (board.side_to_move(1) == Stockfish::BLACK) {
                probs[i] = policyOutput[NUM_POLICY_VALUES + POLICY_INDEX[mirror_move(uci)]]; 
            }
            else {
                probs[i] = policyOutput[NUM_POLICY_VALUES + POLICY_INDEX[uci]];
            }
        }
    }
    return softmax(probs); 
}

inline DynamicVector<float> dynamic_vector(float* p, size_t length) {
    DynamicVector<float> v(length);
    for(size_t i = 0; i < length; i++) {
        v[i] = p[i]; 
    }
    return v; 
}

inline DynamicVector<float> greater_than(DynamicVector<float> v, float threshold) {
    for(size_t i = 0; i < v.size(); i++) {
        v[i] = v[i] > threshold ? 1 : 0; 
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