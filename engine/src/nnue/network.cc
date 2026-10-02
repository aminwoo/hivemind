#include "nnue/network.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace nnue {

namespace {

constexpr char MAGIC_V1[8] = {'H', 'M', 'N', 'N', 'U', 'E', '0', '1'};
constexpr char MAGIC_V2[8] = {'H', 'M', 'N', 'N', 'U', 'E', '0', '2'};
constexpr int KING_LOCAL_BEGIN = 5 * 64;  // own king block inside a board
constexpr int KING_LOCAL_END = 6 * 64;

template <typename T>
bool read_array(std::ifstream& stream, std::vector<T>& out, size_t count) {
    out.resize(count);
    stream.read(reinterpret_cast<char*>(out.data()),
                static_cast<std::streamsize>(count * sizeof(T)));
    return static_cast<bool>(stream);
}

int words_for(int features) {
    return (features + 63) / 64;
}

/// mirror_feature() as a table: it is applied to every feature of every move.
const std::array<uint16_t, NUM_FEATURES>& mirror_table() {
    static const std::array<uint16_t, NUM_FEATURES> table = [] {
        std::array<uint16_t, NUM_FEATURES> result{};
        for (int f = 0; f < NUM_FEATURES; ++f) {
            result[f] = static_cast<uint16_t>(mirror_feature(f));
        }
        return result;
    }();
    return table;
}

inline float screlu(float x) {
    x = std::clamp(x, 0.0f, 1.0f);
    return x * x;
}

#if defined(__AVX2__)
/// For each 8-bit mask, the positions of its set bits, packed first. Lets the
/// forward pass append a chunk's non-zero indices without branching.
struct alignas(64) MaskIndexTable {
    std::array<std::array<uint16_t, 8>, 256> offsets{};
    MaskIndexTable() {
        for (int mask = 0; mask < 256; ++mask) {
            int count = 0;
            for (int bit = 0; bit < 8; ++bit) {
                if (mask & (1 << bit)) {
                    offsets[mask][count++] = static_cast<uint16_t>(bit);
                }
            }
        }
    }
};
const MaskIndexTable MASK_INDICES;

// The kernels below keep their sums in R registers. R is a template
// parameter so the compiler can unroll and hold them in registers; with a
// runtime count they are spilled to memory on every add.

/// out[offset, offset + 16R) = source + added rows - removed rows.
template <int R>
inline void apply_rows(const int16_t* source, int16_t* out, const int16_t* weights, int hidden,
                       const uint16_t* added, int addedCount,
                       const uint16_t* removed, int removedCount, int offset) {
    __m256i sum[R];
#pragma GCC unroll 4
    for (int r = 0; r < R; ++r) {
        sum[r] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source + offset + r * 16));
    }
    for (int k = 0; k < addedCount; ++k) {
        const int16_t* row = weights + static_cast<size_t>(added[k]) * hidden + offset;
#pragma GCC unroll 4
        for (int r = 0; r < R; ++r) {
            sum[r] = _mm256_add_epi16(
                sum[r], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + r * 16)));
        }
    }
    for (int k = 0; k < removedCount; ++k) {
        const int16_t* row = weights + static_cast<size_t>(removed[k]) * hidden + offset;
#pragma GCC unroll 4
        for (int r = 0; r < R; ++r) {
            sum[r] = _mm256_sub_epi16(
                sum[r], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + r * 16)));
        }
    }
#pragma GCC unroll 4
    for (int r = 0; r < R; ++r) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(out + offset + r * 16), sum[r]);
    }
}

#if defined(__FMA__)
/// out[block, block + 8R) += sum over active inputs of input * transposed row.
template <int R>
inline void sparse_block(const float* input, const uint16_t* active, int activeCount,
                         const float* transposed, int width, int block, float* out) {
    // Four independent sets of sums: one chain of dependent FMAs would be
    // bound by their latency rather than throughput.
    __m256 sum[4][R];
#pragma GCC unroll 4
    for (int r = 0; r < R; ++r) {
        sum[0][r] = _mm256_loadu_ps(out + block + r * 8);
        sum[1][r] = _mm256_setzero_ps();
        sum[2][r] = _mm256_setzero_ps();
        sum[3][r] = _mm256_setzero_ps();
    }
    int k = 0;
    for (; k + 4 <= activeCount; k += 4) {
#pragma GCC unroll 4
        for (int lane = 0; lane < 4; ++lane) {
            const int index = active[k + lane];
            const __m256 value = _mm256_set1_ps(input[index]);
            const float* row = transposed + static_cast<size_t>(index) * width + block;
#pragma GCC unroll 4
            for (int r = 0; r < R; ++r) {
                sum[lane][r] = _mm256_fmadd_ps(value, _mm256_loadu_ps(row + r * 8), sum[lane][r]);
            }
        }
    }
    for (; k < activeCount; ++k) {
        const int index = active[k];
        const __m256 value = _mm256_set1_ps(input[index]);
        const float* row = transposed + static_cast<size_t>(index) * width + block;
#pragma GCC unroll 4
        for (int r = 0; r < R; ++r) {
            sum[0][r] = _mm256_fmadd_ps(value, _mm256_loadu_ps(row + r * 8), sum[0][r]);
        }
    }
#pragma GCC unroll 4
    for (int r = 0; r < R; ++r) {
        const __m256 total = _mm256_add_ps(_mm256_add_ps(sum[0][r], sum[1][r]),
                                           _mm256_add_ps(sum[2][r], sum[3][r]));
        _mm256_storeu_ps(out + block + r * 8, total);
    }
}
#endif
#endif

}  // namespace

bool Network::load(const std::string& path, std::string* error) {
    auto fail = [&](const std::string& message) {
        if (error) {
            *error = message;
        }
        hidden_ = 0;
        return false;
    };
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return fail("cannot open " + path);
    }
    char magic[8];
    stream.read(magic, sizeof(magic));
    const bool v1 = stream && std::memcmp(magic, MAGIC_V1, sizeof(magic)) == 0;
    const bool v2 = stream && std::memcmp(magic, MAGIC_V2, sizeof(magic)) == 0;
    if (!v1 && !v2) {
        return fail(path + " is not an HMNNUE network");
    }
    // v1: features, hidden, l1, l2, scale. v2 adds the king bucket count and
    // a 64-entry bucket table.
    uint32_t header[6] = {0, 0, 0, 0, 0, 1};
    stream.read(reinterpret_cast<char*>(header), (v2 ? 6 : 5) * sizeof(uint32_t));
    const auto [features, hidden, l1, l2, scale, buckets] =
        std::tuple{header[0], header[1], header[2], header[3], header[4], header[5]};
    if (!stream || buckets == 0 || buckets > MAX_KING_BUCKETS) {
        return fail("unsupported king bucket count " + std::to_string(buckets));
    }
    if (v2) {
        stream.read(reinterpret_cast<char*>(bucketTable_.data()), 64);
    } else {
        bucketTable_.fill(0);
    }
    const uint32_t expected = 2 * buckets * PIECE_FEATURES_PER_BOARD
        + (NUM_FEATURES - PIECE_FEATURES);
    if (!stream || features != expected) {
        return fail("network has " + std::to_string(features) + " features; engine expects "
                    + std::to_string(expected) + " for " + std::to_string(buckets) + " king buckets");
    }
    for (uint8_t bucket : bucketTable_) {
        if (bucket >= buckets) {
            return fail("king bucket table refers to a missing bucket");
        }
    }
    if (hidden == 0 || hidden > MAX_HIDDEN || hidden % 16 != 0) {
        return fail("unsupported hidden size " + std::to_string(hidden));
    }
    if ((l1 == 0) != (l2 == 0) || l1 > 256 || l2 > 256) {
        return fail("unsupported head " + std::to_string(l1) + "x" + std::to_string(l2));
    }
    const size_t inputs = 2 * static_cast<size_t>(hidden);
    std::vector<float> outBias;
    bool ok = read_array(stream, ftWeights_, static_cast<size_t>(features) * hidden)
        && read_array(stream, ftBias_, hidden);
    if (l1 > 0) {
        ok = ok && read_array(stream, fc1Weights_, static_cast<size_t>(l1) * inputs)
            && read_array(stream, fc1Bias_, l1)
            && read_array(stream, fc2Weights_, static_cast<size_t>(l2) * l1)
            && read_array(stream, fc2Bias_, l2)
            && read_array(stream, outWeights_, l2);
    } else {
        ok = ok && read_array(stream, outWeights_, inputs);
    }
    ok = ok && read_array(stream, outBias, 1);
    if (!ok || stream.peek() != std::char_traits<char>::eof()) {
        return fail(path + " is truncated or has trailing data");
    }
    outBias_ = outBias[0];
    // The forward pass reads the first two layers input-major.
    fc1Transposed_.assign(static_cast<size_t>(l1) * inputs, 0.0f);
    for (size_t j = 0; j < l1; ++j) {
        for (size_t i = 0; i < inputs; ++i) {
            fc1Transposed_[i * l1 + j] = fc1Weights_[j * inputs + i];
        }
    }
    fc2Transposed_.assign(static_cast<size_t>(l2) * l1, 0.0f);
    for (size_t k = 0; k < l2; ++k) {
        for (size_t j = 0; j < l1; ++j) {
            fc2Transposed_[j * l2 + k] = fc2Weights_[k * l1 + j];
        }
    }
    hidden_ = static_cast<int>(hidden);
    l1_ = static_cast<int>(l1);
    l2_ = static_cast<int>(l2);
    features_ = static_cast<int>(features);
    kingBuckets_ = static_cast<int>(buckets);
    ftScale_ = static_cast<float>(scale);
    return true;
}

void Network::team_features(Board& board, bool whiteTeamHasTimeAdvantage,
                            std::array<FeatureSet, 2>& sets) const {
    std::array<uint16_t, MAX_ACTIVE_FEATURES> base;
    const int count = extract_features(
        board, Stockfish::WHITE, whiteTeamHasTimeAdvantage, base.data());
    const std::array<uint16_t, NUM_FEATURES>& mirror = mirror_table();
    const int words = words_for(features_);
    std::fill_n(sets[0].begin(), words, 0);
    std::fill_n(sets[1].begin(), words, 0);
    auto set_bit = [](FeatureSet& set, int f) {
        set[f >> 6] |= uint64_t{1} << (f & 63);
    };
    if (kingBuckets_ == 1) {
        // Without king buckets the network features are the plain ones, and
        // the black team's are the mirror of the white team's.
        for (int i = 0; i < count; ++i) {
            set_bit(sets[0], base[i]);
            set_bit(sets[1], mirror[base[i]]);
        }
        return;
    }
    for (int team = 0; team < 2; ++team) {
        std::array<int, MAX_ACTIVE_FEATURES> list;
        std::array<int, 2> bucket = {0, 0};
        for (int i = 0; i < count; ++i) {
            list[i] = team == 0 ? base[i] : mirror[base[i]];
            const int local = list[i] % PIECE_FEATURES_PER_BOARD;
            if (list[i] < PIECE_FEATURES && local >= KING_LOCAL_BEGIN && local < KING_LOCAL_END) {
                bucket[list[i] / PIECE_FEATURES_PER_BOARD] = bucketTable_[local - KING_LOCAL_BEGIN];
            }
        }
        for (int i = 0; i < count; ++i) {
            const int f = list[i];
            if (f < PIECE_FEATURES) {
                const int boardNumber = f / PIECE_FEATURES_PER_BOARD;
                set_bit(sets[team], (boardNumber * kingBuckets_ + bucket[boardNumber])
                                        * PIECE_FEATURES_PER_BOARD + f % PIECE_FEATURES_PER_BOARD);
            } else {
                set_bit(sets[team], 2 * kingBuckets_ * PIECE_FEATURES_PER_BOARD + (f - PIECE_FEATURES));
            }
        }
    }
}

void Network::add_row(int16_t* values, int feature) const {
    const int16_t* row = ftWeights_.data() + static_cast<size_t>(feature) * hidden_;
#if defined(__AVX2__)
    for (int i = 0; i < hidden_; i += 16) {
        __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(values + i));
        __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(values + i), _mm256_add_epi16(v, w));
    }
#else
    for (int i = 0; i < hidden_; ++i) {
        values[i] = static_cast<int16_t>(values[i] + row[i]);
    }
#endif
}

void Network::sub_row(int16_t* values, int feature) const {
    const int16_t* row = ftWeights_.data() + static_cast<size_t>(feature) * hidden_;
#if defined(__AVX2__)
    for (int i = 0; i < hidden_; i += 16) {
        __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(values + i));
        __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(values + i), _mm256_sub_epi16(v, w));
    }
#else
    for (int i = 0; i < hidden_; ++i) {
        values[i] = static_cast<int16_t>(values[i] - row[i]);
    }
#endif
}

void Network::refresh(Board& board, bool whiteTeamHasTimeAdvantage, Accumulator& acc) const {
    team_features(board, whiteTeamHasTimeAdvantage, acc.features);
    const int words = words_for(features_);
    for (int team = 0; team < 2; ++team) {
        int16_t* values = acc.values[team].data();
        std::copy_n(ftBias_.data(), hidden_, values);
        for (int word = 0; word < words; ++word) {
            uint64_t bits = acc.features[team][word];
            while (bits) {
                add_row(values, word * 64 + std::countr_zero(bits));
                bits &= bits - 1;
            }
        }
    }
}

void Network::update(Board& board, bool whiteTeamHasTimeAdvantage,
                     const Accumulator& parent, Accumulator& acc) const {
    team_features(board, whiteTeamHasTimeAdvantage, acc.features);
    const int words = words_for(features_);
    for (int team = 0; team < 2; ++team) {
        const FeatureSet& now = acc.features[team];
        const FeatureSet& before = parent.features[team];
        int16_t* values = acc.values[team].data();
        std::array<uint16_t, 2 * MAX_ACTIVE_FEATURES> added;
        std::array<uint16_t, 2 * MAX_ACTIVE_FEATURES> removed;
        int addedCount = 0;
        int removedCount = 0;
        for (int word = 0; word < words; ++word) {
            uint64_t diff = now[word] ^ before[word];
            while (diff) {
                const int bit = std::countr_zero(diff);
                diff &= diff - 1;
                const auto feature = static_cast<uint16_t>(word * 64 + bit);
                if (now[word] & (uint64_t{1} << bit)) {
                    added[addedCount++] = feature;
                } else {
                    removed[removedCount++] = feature;
                }
            }
        }
        // Past about half the active set (a king changing bucket moves a
        // whole board) a refresh is cheaper than a diff.
        const bool refreshTeam = addedCount + removedCount > 40;
        const int16_t* source = refreshTeam ? ftBias_.data() : parent.values[team].data();
        if (refreshTeam) {
            addedCount = 0;
            removedCount = 0;
            for (int word = 0; word < words; ++word) {
                uint64_t bits = now[word];
                while (bits) {
                    added[addedCount++] = static_cast<uint16_t>(word * 64 + std::countr_zero(bits));
                    bits &= bits - 1;
                }
            }
        }
        // One pass per block: the parent (or bias) is read and the result
        // written once, with every changed row applied in registers.
#if defined(__AVX2__)
        int offset = 0;
        for (; offset + 64 <= hidden_; offset += 64) {
            apply_rows<4>(source, values, ftWeights_.data(), hidden_,
                          added.data(), addedCount, removed.data(), removedCount, offset);
        }
        for (; offset < hidden_; offset += 16) {
            apply_rows<1>(source, values, ftWeights_.data(), hidden_,
                          added.data(), addedCount, removed.data(), removedCount, offset);
        }
#else
        std::copy_n(source, hidden_, values);
        for (int k = 0; k < addedCount; ++k) {
            add_row(values, added[k]);
        }
        for (int k = 0; k < removedCount; ++k) {
            sub_row(values, removed[k]);
        }
#endif
    }
}

float Network::forward(const Accumulator& acc, Stockfish::Color team) const {
    // SCReLU zeroes about two thirds of the accumulator, so the first layer
    // runs over the non-zero inputs only: each adds its value times one row
    // of the transposed weights to every output at once.
    alignas(32) std::array<float, 2 * MAX_HIDDEN> input;
    std::array<uint16_t, 2 * MAX_HIDDEN + 8> active;  // slack for 8-wide stores
    int activeCount = 0;
    const int us = team == Stockfish::WHITE ? 0 : 1;
    const float inverseScale = 1.0f / ftScale_;
    for (int side = 0; side < 2; ++side) {
        const int16_t* values = acc.values[side == 0 ? us : 1 - us].data();
        float* out = input.data() + side * hidden_;
        const int base = side * hidden_;
#if defined(__AVX2__)
        const __m256 scale = _mm256_set1_ps(inverseScale);
        const __m256 zero = _mm256_setzero_ps();
        const __m256 one = _mm256_set1_ps(1.0f);
        for (int i = 0; i < hidden_; i += 8) {
            const __m128i raw = _mm_load_si128(reinterpret_cast<const __m128i*>(values + i));
            __m256 x = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(raw)), scale);
            x = _mm256_min_ps(_mm256_max_ps(x, zero), one);
            x = _mm256_mul_ps(x, x);
            _mm256_store_ps(out + i, x);
            const unsigned mask = static_cast<unsigned>(
                _mm256_movemask_ps(_mm256_cmp_ps(x, zero, _CMP_GT_OQ)));
            // Write all eight candidate indices, keep popcount of them.
            const __m128i offsets = _mm_load_si128(
                reinterpret_cast<const __m128i*>(MASK_INDICES.offsets[mask].data()));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(active.data() + activeCount),
                             _mm_add_epi16(offsets, _mm_set1_epi16(static_cast<short>(base + i))));
            activeCount += std::popcount(mask);
        }
#else
        for (int i = 0; i < hidden_; ++i) {
            out[i] = screlu(static_cast<float>(values[i]) * inverseScale);
            if (out[i] > 0.0f) {
                active[activeCount++] = static_cast<uint16_t>(base + i);
            }
        }
#endif
    }
    if (l1_ == 0) {
        float total = outBias_;
        for (int k = 0; k < activeCount; ++k) {
            total += outWeights_[active[k]] * input[active[k]];
        }
        return total;
    }

    alignas(32) std::array<float, 256> hidden1;
    alignas(32) std::array<float, 256> hidden2;
    std::copy_n(fc1Bias_.data(), l1_, hidden1.data());
#if defined(__AVX2__) && defined(__FMA__)
    if (l1_ % 8 == 0) {
        const float* transposed = fc1Transposed_.data();
        int block = 0;
        for (; block + 32 <= l1_; block += 32) {
            sparse_block<4>(input.data(), active.data(), activeCount, transposed, l1_, block, hidden1.data());
        }
        for (; block + 16 <= l1_; block += 16) {
            sparse_block<2>(input.data(), active.data(), activeCount, transposed, l1_, block, hidden1.data());
        }
        for (; block < l1_; block += 8) {
            sparse_block<1>(input.data(), active.data(), activeCount, transposed, l1_, block, hidden1.data());
        }
    } else
#endif
    {
        for (int k = 0; k < activeCount; ++k) {
            const float value = input[active[k]];
            const float* row = fc1Transposed_.data() + static_cast<size_t>(active[k]) * l1_;
            for (int j = 0; j < l1_; ++j) {
                hidden1[j] += value * row[j];
            }
        }
    }
    for (int j = 0; j < l1_; ++j) {
        hidden1[j] = screlu(hidden1[j]);
    }
    std::copy_n(fc2Bias_.data(), l2_, hidden2.data());
    for (int j = 0; j < l1_; ++j) {
        const float value = hidden1[j];
        if (value == 0.0f) {
            continue;
        }
        const float* row = fc2Transposed_.data() + static_cast<size_t>(j) * l2_;
        for (int k = 0; k < l2_; ++k) {
            hidden2[k] += value * row[k];
        }
    }
    float total = outBias_;
    for (int k = 0; k < l2_; ++k) {
        total += outWeights_[k] * screlu(hidden2[k]);
    }
    return total;
}

float Network::evaluate(Board& board, Stockfish::Color team, bool teamHasTimeAdvantage) const {
    Accumulator acc;
    const bool whiteAdvantage = team == Stockfish::WHITE
        ? teamHasTimeAdvantage : !teamHasTimeAdvantage;
    refresh(board, whiteAdvantage, acc);
    return forward(acc, team);
}

}  // namespace nnue
