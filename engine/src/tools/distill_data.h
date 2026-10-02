#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "environment/board.h"
#include "environment/constants.h"
#include "nn/backend_compat.h"

/**
 * @brief Network-distillation chunks (HDST), shared by `gennnue --format
 * distill` (raw teacher outputs) and `selfplay --distill-output` (search
 * results). The Python side reads them with hivemind.distill.data.
 *
 * Layout (v2): magic, version, planes per position, N, total policy entries
 * P, then planeBits u64[N*74], planeValues f16[N*74], value f32[N],
 * wdl f32[N*3] (loss, draw, win), movesLeft f16[N], ply u16[N],
 * outcome i8[N], policyCount u16[N*2] (board A, board B; 0 when the board is
 * not on turn), policyIndex u16[P] (every legal move and the pass),
 * policyProbability f16[P].
 *
 * A record whose WDL is all zero has no WDL or moves-left target. Search
 * records store root Q as the value and, when the game finished, its result as
 * a one-hot WDL (zero for games cut off at the macro-ply limit).
 */
namespace distill {

constexpr int8_t OUTCOME_UNKNOWN = 2;

struct PolicyEntry {
    uint16_t index;
    uint16_t probability;  // IEEE half bits
};

struct Record {
    std::array<uint64_t, NB_INPUT_CHANNELS> planeBits{};
    std::array<uint16_t, NB_INPUT_CHANNELS> planeValues{};
    float value = 0.0f;
    std::array<float, 3> wdl{};
    uint16_t movesLeft = 0;  // IEEE half bits
    uint16_t ply = 0;
    int8_t outcome = OUTCOME_UNKNOWN;
    std::array<std::vector<PolicyEntry>, 2> policy;
};

uint16_t half_bits(__half value);

/**
 * @brief Packs input planes exactly: every plane is either one value on all
 * 64 squares (pockets, turn, castling, clocks) or a 0/1 bitboard (pieces, en
 * passant, last move). Anything else throws, so a new plane type cannot be
 * written silently wrong.
 */
void pack_planes(const __half* planes, Record& record);

/// The moves a board's policy target covers: every legal move the policy can
/// represent, then the pass (Stockfish::MOVE_NONE), in the search's order.
std::vector<Stockfish::Move> policy_actions(Board& board, int boardNumber);

/// Writes `records` to `path` atomically (through a temporary file).
void write_chunk(const std::filesystem::path& path, const std::vector<Record>& records);

/// Buffers records and writes `<prefix>_<index>.dst` chunks.
class ChunkWriter {
public:
    ChunkWriter(std::filesystem::path directory, std::string prefix, size_t chunkPositions);
    void append(Record&& record);
    void flush();

private:
    std::filesystem::path directory_;
    std::string prefix_;
    size_t chunkPositions_;
    size_t chunkIndex_ = 0;
    std::vector<Record> records_;
};

}  // namespace distill
