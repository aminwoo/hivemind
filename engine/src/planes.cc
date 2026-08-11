#include "planes.h"
#include <algorithm>
#include <type_traits>

namespace {

constexpr int MAX_NUM_NO_PROGRESS = 50;

}

// Check for AVX2 support at compile time
#if defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

// Stockfish bitboards are typically 64-bit integers where bit 0 is A1.
// This function maps the bitboard to the current plane iterator.
// Optimized: Uses bit-scanning intrinsics for O(popcount) instead of O(64)
template <typename T>
inline T plane_value(float value) {
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half_rn(value);
    }
    return value;
}

template <typename T>
inline void set_bits_from_bitmap(Stockfish::Bitboard bb, T* curIt) {
    while (bb) {
        // __builtin_ctzll: count trailing zeros (index of lowest set bit)
        int idx = __builtin_ctzll(bb);
        curIt[idx] = plane_value<T>(1.0f);
        bb &= bb - 1;  // Clear lowest set bit
    }
}

// SIMD-optimized fill for 64 floats (one chess board plane)
template <typename T>
inline void fill_plane(T* curIt, float value) {
#if USE_AVX2
    if constexpr (std::is_same_v<T, float>) {
        // AVX2: Fill 8 floats at a time (256 bits = 8 x 32-bit floats)
        __m256 val = _mm256_set1_ps(value);
        for (int i = 0; i < 64; i += 8) {
            _mm256_storeu_ps(curIt + i, val);
        }
        return;
    }
#endif
    std::fill_n(curIt, 64, plane_value<T>(value));
}

template <typename T>
struct PlaneData {
    Board& board;
    T* inputPlanes;
    T* curIt;
    Stockfish::Color teamSide; 

    PlaneData(Board& board, T* inputPlanes, Stockfish::Color teamSide)
        : board(board), inputPlanes(inputPlanes), curIt(inputPlanes), teamSide(teamSide) {}

    inline void increment_channel() {
        curIt += 64;
    }

    inline void set_plane_to_value(float value) {
        fill_plane(curIt, value);
        increment_channel();
    }

    inline void set_plane_to_bitboard(Stockfish::Bitboard bb) {
        set_bits_from_bitmap(bb, curIt);
        increment_channel();
    }

    // Helper to determine if the specific board needs perspective flipping
    inline bool needs_flipping(int boardIdx) {
        if (boardIdx == 0) return teamSide == Stockfish::BLACK;
        return teamSide == Stockfish::WHITE;
    }
};

template <typename T>
inline void set_plane_pieces_board(PlaneData<T>& p, int boardIdx) {
    // Python logic: Board A uses (us, opponent), Board B uses (opponent, us)
    Stockfish::Color first = (boardIdx == 0) ? p.teamSide : ~p.teamSide;
    Stockfish::Color second = ~first;

    for (Stockfish::Color color : {first, second}) {
        for (Stockfish::PieceType piece : {Stockfish::PAWN, Stockfish::KNIGHT, Stockfish::BISHOP, Stockfish::ROOK, Stockfish::QUEEN, Stockfish::KING}) {
            Stockfish::Bitboard bb = p.board.pieces(boardIdx, color, piece);
            if (p.needs_flipping(boardIdx)) {
                bb = flip_vertical(bb);
            }
            p.set_plane_to_bitboard(bb);
        }
    }
}

template <typename T>
inline void set_plane_pockets_board(PlaneData<T>& p, int boardIdx) {
    Stockfish::Color first = (boardIdx == 0) ? p.teamSide : ~p.teamSide;
    Stockfish::Color second = ~first;

    for (Stockfish::Color color : {first, second}) {
        for (Stockfish::PieceType piece : {Stockfish::PAWN, Stockfish::KNIGHT, Stockfish::BISHOP, Stockfish::ROOK, Stockfish::QUEEN}) {
            float val = (float)p.board.count_in_hand(boardIdx, color, piece) / 16.0f;
            p.set_plane_to_value(val);
        }
    }
}

template <typename T>
inline void set_plane_promoted_pieces_board(PlaneData<T>& p, int boardIdx) {
    Stockfish::Color first = (boardIdx == 0) ? p.teamSide : ~p.teamSide;
    Stockfish::Color second = ~first;

    for (Stockfish::Color color : {first, second}) {
        Stockfish::Bitboard bb = p.board.promoted_pieces(boardIdx) & p.board.pieces(boardIdx, color); 
        if (p.needs_flipping(boardIdx)) {
            bb = flip_vertical(bb);
        }
        p.set_plane_to_bitboard(bb);
    }
}

template <typename T>
inline void set_plane_ep_square_board(PlaneData<T>& p, int boardIdx) {
    Stockfish::Bitboard bb = 0; 
    auto ep_sq = p.board.ep_square(boardIdx);
    if (ep_sq != Stockfish::SQ_NONE) {
        bb = 1ULL << ep_sq;
    }
    if (p.needs_flipping(boardIdx)) {
        bb = flip_vertical(bb);
    }
    p.set_plane_to_bitboard(bb);
}

template <typename T>
inline void set_plane_color_info_board(PlaneData<T>& p, int boardIdx) {
    // Python: Board A turn == team_side; Board B turn == not team_side
    Stockfish::Color expectedTurn = (boardIdx == 0) ? p.teamSide : ~p.teamSide;
    float val = (p.board.side_to_move(boardIdx) == expectedTurn) ? 1.0f : 0.0f; 
    p.set_plane_to_value(val);
}

template <typename T>
inline void set_plane_castling_rights_board(PlaneData<T>& p, int boardIdx) {
    // Note: Castling rights are usually evaluated on the original board state
    if (boardIdx == 0) {
        p.set_plane_to_value(p.board.can_castle(0, p.teamSide == Stockfish::WHITE ? Stockfish::WHITE_OO : Stockfish::BLACK_OO));
        p.set_plane_to_value(p.board.can_castle(0, p.teamSide == Stockfish::WHITE ? Stockfish::WHITE_OOO : Stockfish::BLACK_OOO));
        p.set_plane_to_value(p.board.can_castle(0, p.teamSide == Stockfish::WHITE ? Stockfish::BLACK_OO : Stockfish::WHITE_OO));
        p.set_plane_to_value(p.board.can_castle(0, p.teamSide == Stockfish::WHITE ? Stockfish::BLACK_OOO : Stockfish::WHITE_OOO));
    } else {
        // Board B logic: Kingside(not team_side), Queenside(not team_side), Kingside(team_side), Queenside(team_side)
        p.set_plane_to_value(p.board.can_castle(1, p.teamSide == Stockfish::WHITE ? Stockfish::BLACK_OO : Stockfish::WHITE_OO));
        p.set_plane_to_value(p.board.can_castle(1, p.teamSide == Stockfish::WHITE ? Stockfish::BLACK_OOO : Stockfish::WHITE_OOO));
        p.set_plane_to_value(p.board.can_castle(1, p.teamSide == Stockfish::WHITE ? Stockfish::WHITE_OO : Stockfish::BLACK_OO));
        p.set_plane_to_value(p.board.can_castle(1, p.teamSide == Stockfish::WHITE ? Stockfish::WHITE_OOO : Stockfish::BLACK_OOO));
    }
}

inline Stockfish::Square orient_square(Stockfish::Square square, bool flip) {
    return flip ? Stockfish::Square(static_cast<int>(square) ^ 56) : square;
}

template <typename T>
inline void set_plane_last_move_board(PlaneData<T>& p, int boardIdx) {
    const Stockfish::Move move = p.board.last_move(boardIdx);
    if (move == Stockfish::MOVE_NONE) {
        p.increment_channel();
        p.increment_channel();
        return;
    }

    const bool flip = p.needs_flipping(boardIdx);
    if (Stockfish::type_of(move) != Stockfish::DROP) {
        p.curIt[orient_square(Stockfish::from_sq(move), flip)] = plane_value<T>(1.0f);
    }
    p.increment_channel();
    p.curIt[orient_square(Stockfish::to_sq(move), flip)] = plane_value<T>(1.0f);
    p.increment_channel();
}

template <typename T>
inline void set_plane_history_board(PlaneData<T>& p, int boardIdx) {
    set_plane_last_move_board(p, boardIdx);
    const float halfmoveClock = std::min(
        p.board.rule50_count(boardIdx), MAX_NUM_NO_PROGRESS
    ) / static_cast<float>(MAX_NUM_NO_PROGRESS);
    p.set_plane_to_value(halfmoveClock);

    const int repetitions = p.board.repetition_count(boardIdx);
    p.set_plane_to_value(repetitions >= 2 ? 1.0f : 0.0f);
    p.set_plane_to_value(repetitions >= 3 ? 1.0f : 0.0f);
}

template <typename T>
void board_to_planes_impl(Board& board, T* inputPlanes, Stockfish::Color teamSide, bool hasTimeAdvantage) {
    // Initialize all to 0 using SIMD when available
    constexpr size_t totalValues = NB_INPUT_VALUES();
#if USE_AVX2
    if constexpr (std::is_same_v<T, float>) {
        __m256 zero = _mm256_setzero_ps();
        for (size_t i = 0; i < totalValues; i += 8) {
            _mm256_storeu_ps(inputPlanes + i, zero);
        }
    } else {
        std::fill_n(inputPlanes, totalValues, plane_value<T>(0.0f));
    }
#else
    std::fill_n(inputPlanes, totalValues, plane_value<T>(0.0f));
#endif
    PlaneData<T> planeData(board, inputPlanes, teamSide);
    
    // Process Board 0 (Channels 0-31)
    set_plane_pieces_board(planeData, 0);           
    set_plane_pockets_board(planeData, 0);          
    set_plane_promoted_pieces_board(planeData, 0);  
    set_plane_ep_square_board(planeData, 0);        
    set_plane_color_info_board(planeData, 0);       
    planeData.set_plane_to_value(1.0f);             // Constant plane
    set_plane_castling_rights_board(planeData, 0);  
    planeData.set_plane_to_value(hasTimeAdvantage ? 1.0f : 0.0f); 
    set_plane_history_board(planeData, 0);
    
    // Process Board 1
    set_plane_pieces_board(planeData, 1);           
    set_plane_pockets_board(planeData, 1);          
    set_plane_promoted_pieces_board(planeData, 1);  
    set_plane_ep_square_board(planeData, 1);        
    set_plane_color_info_board(planeData, 1);       
    planeData.set_plane_to_value(1.0f);             // Constant plane
    set_plane_castling_rights_board(planeData, 1);  
    planeData.set_plane_to_value(hasTimeAdvantage ? 1.0f : 0.0f);
    set_plane_history_board(planeData, 1);
}

void board_to_planes(Board& board, float* inputPlanes, Stockfish::Color teamSide, bool hasTimeAdvantage) {
    board_to_planes_impl(board, inputPlanes, teamSide, hasTimeAdvantage);
}

void board_to_planes(Board& board, __half* inputPlanes, Stockfish::Color teamSide, bool hasTimeAdvantage) {
    board_to_planes_impl(board, inputPlanes, teamSide, hasTimeAdvantage);
}