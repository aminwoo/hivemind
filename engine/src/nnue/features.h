#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "environment/board.h"
#include "Fairy-Stockfish/src/types.h"

/**
 * @brief Sparse NNUE input features for a bughouse position.
 *
 * Features are defined from the point of view of one team. The team plays its
 * own colour on board A and the opposite colour on board B; every board is
 * oriented so the team's pieces start at the bottom, matching the network
 * planes. Each feature has a "relation": own (0) or opponent (1).
 *
 * The definition is symmetric, so the other team's features are a fixed
 * permutation of these: flip the relation and mirror squares vertically.
 * mirror_feature() implements that permutation, and the training data stores
 * only the side-to-move list.
 *
 *   PIECE     board x rel x piece(6) x square(64)          1536
 *   HAND      board x rel x thermometer(P16 N4 B4 R4 Q2)    120
 *   PROMOTED  board x rel x square                          256
 *   ON_TURN   board x rel (rel's colour to move there)        4
 *   TIME_ADV  rel (that team holds the time advantage)        2
 *   CASTLE    board x rel x {OO, OOO}                         8
 */
namespace nnue {

constexpr int PIECE_TYPES = 6;
constexpr int HAND_PIECE_TYPES = 5;
constexpr std::array<int, HAND_PIECE_TYPES> HAND_CAPS = {16, 4, 4, 4, 2};
constexpr std::array<int, HAND_PIECE_TYPES> HAND_OFFSETS = {0, 16, 20, 24, 28};
constexpr int HAND_SLOTS = 30;

constexpr int PIECE_OFFSET = 0;
constexpr int PIECE_FEATURES = 2 * 2 * PIECE_TYPES * 64;
constexpr int HAND_OFFSET = PIECE_OFFSET + PIECE_FEATURES;
constexpr int HAND_FEATURES = 2 * 2 * HAND_SLOTS;
constexpr int PROMOTED_OFFSET = HAND_OFFSET + HAND_FEATURES;
constexpr int PROMOTED_FEATURES = 2 * 2 * 64;
constexpr int ON_TURN_OFFSET = PROMOTED_OFFSET + PROMOTED_FEATURES;
constexpr int ON_TURN_FEATURES = 2 * 2;
constexpr int TIME_ADV_OFFSET = ON_TURN_OFFSET + ON_TURN_FEATURES;
constexpr int TIME_ADV_FEATURES = 2;
constexpr int CASTLE_OFFSET = TIME_ADV_OFFSET + TIME_ADV_FEATURES;
constexpr int CASTLE_FEATURES = 2 * 2 * 2;
constexpr int NUM_FEATURES = CASTLE_OFFSET + CASTLE_FEATURES;

// Pieces on both boards plus pieces in hand never exceed the two sets of 32;
// promoted markers, turn, time and castling flags add at most 32 + 7.
constexpr int MAX_ACTIVE_FEATURES = 128;

constexpr std::array<Stockfish::PieceType, PIECE_TYPES> PIECE_ORDER = {
    Stockfish::PAWN, Stockfish::KNIGHT, Stockfish::BISHOP,
    Stockfish::ROOK, Stockfish::QUEEN, Stockfish::KING};

/// Colour `team` plays on `board`.
inline Stockfish::Color own_color(int board, Stockfish::Color team) {
    return board == 0 ? team : ~team;
}

inline int orient(int square, Stockfish::Color ownColor) {
    return ownColor == Stockfish::BLACK ? square ^ 56 : square;
}

inline int piece_feature(int board, int rel, int piece, int square) {
    return PIECE_OFFSET + ((board * 2 + rel) * PIECE_TYPES + piece) * 64 + square;
}

inline int hand_feature(int board, int rel, int slot) {
    return HAND_OFFSET + (board * 2 + rel) * HAND_SLOTS + slot;
}

inline int promoted_feature(int board, int rel, int square) {
    return PROMOTED_OFFSET + (board * 2 + rel) * 64 + square;
}

/**
 * @brief Writes the active features of `board` seen by `team`.
 *
 * @param teamHasTimeAdvantage Whether `team` holds the time advantage.
 * @return The number of indices written to `out` (at most MAX_ACTIVE_FEATURES).
 */
inline int extract_features(Board& board, Stockfish::Color team,
                            bool teamHasTimeAdvantage, uint16_t* out) {
    int count = 0;
    for (int b = 0; b < 2; ++b) {
        const Stockfish::Color ownColor = own_color(b, team);
        for (int rel = 0; rel < 2; ++rel) {
            const Stockfish::Color color = rel == 0 ? ownColor : ~ownColor;
            for (int p = 0; p < PIECE_TYPES; ++p) {
                Stockfish::Bitboard bb = board.pieces(b, color, PIECE_ORDER[p]);
                while (bb) {
                    const int sq = static_cast<int>(Stockfish::pop_lsb(bb));
                    out[count++] = static_cast<uint16_t>(
                        piece_feature(b, rel, p, orient(sq, ownColor)));
                }
            }
            for (int p = 0; p < HAND_PIECE_TYPES; ++p) {
                const int held = std::min(
                    board.count_in_hand(b, color, PIECE_ORDER[p]), HAND_CAPS[p]);
                for (int k = 0; k < held; ++k) {
                    out[count++] = static_cast<uint16_t>(
                        hand_feature(b, rel, HAND_OFFSETS[p] + k));
                }
            }
            Stockfish::Bitboard promoted =
                board.promoted_pieces(b) & board.pieces(b, color);
            while (promoted) {
                const int sq = static_cast<int>(Stockfish::pop_lsb(promoted));
                out[count++] = static_cast<uint16_t>(
                    promoted_feature(b, rel, orient(sq, ownColor)));
            }
        }
        const int moverRel = board.side_to_move(b) == ownColor ? 0 : 1;
        out[count++] = static_cast<uint16_t>(ON_TURN_OFFSET + b * 2 + moverRel);
    }
    out[count++] = static_cast<uint16_t>(
        TIME_ADV_OFFSET + (teamHasTimeAdvantage ? 0 : 1));
    for (int b = 0; b < 2; ++b) {
        const Stockfish::Color ownColor = own_color(b, team);
        for (int rel = 0; rel < 2; ++rel) {
            const Stockfish::Color color = rel == 0 ? ownColor : ~ownColor;
            const Stockfish::CastlingRights oo = color == Stockfish::WHITE
                ? Stockfish::WHITE_OO : Stockfish::BLACK_OO;
            const Stockfish::CastlingRights ooo = color == Stockfish::WHITE
                ? Stockfish::WHITE_OOO : Stockfish::BLACK_OOO;
            if (board.can_castle(b, oo)) {
                out[count++] = static_cast<uint16_t>(
                    CASTLE_OFFSET + (b * 2 + rel) * 2 + 0);
            }
            if (board.can_castle(b, ooo)) {
                out[count++] = static_cast<uint16_t>(
                    CASTLE_OFFSET + (b * 2 + rel) * 2 + 1);
            }
        }
    }
    return count;
}

/// The same feature seen by the other team: relation flips, squares mirror.
constexpr int mirror_feature(int f) {
    if (f < HAND_OFFSET) {
        const int square = f % 64;
        const int piece = (f / 64) % PIECE_TYPES;
        const int boardRel = f / (64 * PIECE_TYPES);
        return piece_feature(boardRel / 2, (boardRel % 2) ^ 1, piece, square ^ 56);
    }
    if (f < PROMOTED_OFFSET) {
        const int local = f - HAND_OFFSET;
        const int boardRel = local / HAND_SLOTS;
        return hand_feature(boardRel / 2, (boardRel % 2) ^ 1, local % HAND_SLOTS);
    }
    if (f < ON_TURN_OFFSET) {
        const int local = f - PROMOTED_OFFSET;
        const int boardRel = local / 64;
        return promoted_feature(boardRel / 2, (boardRel % 2) ^ 1, (local % 64) ^ 56);
    }
    if (f < TIME_ADV_OFFSET) {
        return f ^ 1;
    }
    if (f < CASTLE_OFFSET) {
        return TIME_ADV_OFFSET + ((f - TIME_ADV_OFFSET) ^ 1);
    }
    const int local = f - CASTLE_OFFSET;
    const int side = local % 2;
    const int boardRel = local / 2;
    return CASTLE_OFFSET + ((boardRel ^ 1) * 2) + side;
}

/// The same feature after swapping boards A and B, which bughouse's rules
/// treat identically. The team keeps its pieces but now counts as the other
/// colour, so only the board index changes. Used for training augmentation.
constexpr int swap_boards_feature(int f) {
    constexpr int PIECE_BOARD_STRIDE = 2 * PIECE_TYPES * 64;
    constexpr int HAND_BOARD_STRIDE = 2 * HAND_SLOTS;
    if (f < HAND_OFFSET) {
        return f < PIECE_BOARD_STRIDE ? f + PIECE_BOARD_STRIDE : f - PIECE_BOARD_STRIDE;
    }
    if (f < PROMOTED_OFFSET) {
        const int local = f - HAND_OFFSET;
        return HAND_OFFSET
            + (local < HAND_BOARD_STRIDE ? local + HAND_BOARD_STRIDE : local - HAND_BOARD_STRIDE);
    }
    if (f < ON_TURN_OFFSET) {
        return PROMOTED_OFFSET + ((f - PROMOTED_OFFSET) ^ (2 * 64));
    }
    if (f < TIME_ADV_OFFSET) {
        return ON_TURN_OFFSET + ((f - ON_TURN_OFFSET) ^ 2);
    }
    if (f < CASTLE_OFFSET) {
        return f;
    }
    return CASTLE_OFFSET + ((f - CASTLE_OFFSET) ^ 4);
}

}  // namespace nnue
