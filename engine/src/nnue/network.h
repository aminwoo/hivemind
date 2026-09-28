#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "environment/board.h"
#include "nnue/features.h"
#include "Fairy-Stockfish/src/types.h"

namespace nnue {

constexpr int MAX_HIDDEN = 1024;
constexpr int MAX_KING_BUCKETS = 16;
constexpr int PIECE_FEATURES_PER_BOARD = 2 * PIECE_TYPES * 64;
constexpr int MAX_NETWORK_FEATURES =
    2 * MAX_KING_BUCKETS * PIECE_FEATURES_PER_BOARD + (NUM_FEATURES - PIECE_FEATURES);
constexpr int FEATURE_WORDS = (MAX_NETWORK_FEATURES + 63) / 64;

using FeatureSet = std::array<uint64_t, FEATURE_WORDS>;

/**
 * @brief Both teams' accumulators for one position.
 *
 * `features[team]` is that team's active network features as a bitset, so a
 * child is built from its parent by adding and removing only the rows whose
 * bits differ.
 */
struct alignas(64) Accumulator {
    std::array<std::array<int16_t, MAX_HIDDEN>, 2> values;  // [team colour][hidden]
    std::array<FeatureSet, 2> features{};
};

/**
 * @brief The distilled evaluation network (see src/hivemind/nnue/model.py).
 *
 * The feature transformer is int16 scaled by `ftScale`; the head runs in
 * float on SCReLU activations. forward() returns the network's logit for the
 * team to play: tanh(logit) estimates the teacher's value.
 *
 * With king buckets, each board's piece features are indexed by the bucket
 * of the perspective team's own king on that board; the remaining features
 * are shared. One bucket reproduces the plain feature set.
 *
 * Which team holds the time advantage is part of the features. It is fixed
 * for a whole search, so callers pass the white team's flag with every
 * refresh or update.
 */
class Network {
public:
    bool load(const std::string& path, std::string* error = nullptr);
    bool loaded() const { return hidden_ > 0; }
    int hidden() const { return hidden_; }
    int king_buckets() const { return kingBuckets_; }

    void refresh(Board& board, bool whiteTeamHasTimeAdvantage, Accumulator& acc) const;
    /// Builds `acc` for `board` from the accumulator of its parent position.
    void update(Board& board, bool whiteTeamHasTimeAdvantage,
                const Accumulator& parent, Accumulator& acc) const;
    float forward(const Accumulator& acc, Stockfish::Color team) const;

    /// One-shot evaluation, for tests and the root.
    float evaluate(Board& board, Stockfish::Color team, bool teamHasTimeAdvantage) const;

private:
    void team_features(Board& board, bool whiteTeamHasTimeAdvantage,
                       std::array<FeatureSet, 2>& sets) const;
    void add_row(int16_t* values, int feature) const;
    void sub_row(int16_t* values, int feature) const;

    int hidden_ = 0;
    int l1_ = 0;
    int l2_ = 0;
    int features_ = 0;
    int kingBuckets_ = 1;
    std::array<uint8_t, 64> bucketTable_{};
    float ftScale_ = 255.0f;
    std::vector<int16_t> ftWeights_;  // [feature][hidden]
    std::vector<int16_t> ftBias_;
    std::vector<float> fc1Weights_, fc1Bias_, fc2Weights_, fc2Bias_, outWeights_;
    std::vector<float> fc1Transposed_;  // [input][l1]
    std::vector<float> fc2Transposed_;  // [l1][l2]
    float outBias_ = 0.0f;
};

}  // namespace nnue
