#include "planes.h"
#include <algorithm>

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
inline void set_bits_from_bitmap(Stockfish::Bitboard bb, float *curIt) {
    while (bb) {
        // __builtin_ctzll: count trailing zeros (index of lowest set bit)
        int idx = __builtin_ctzll(bb);
        curIt[idx] = 1.0f;
        bb &= bb - 1;  // Clear lowest set bit
    }
}

// SIMD-optimized fill for 64 floats (one chess board plane)
inline void fill_plane_simd(float* curIt, float value) {
#if USE_AVX2
    // AVX2: Fill 8 floats at a time (256 bits = 8 x 32-bit floats)
    __m256 val = _mm256_set1_ps(value);
    for (int i = 0; i < 64; i += 8) {
        _mm256_storeu_ps(curIt + i, val);
    }
#else
    // Fallback to std::fill_n
    std::fill_n(curIt, 64, value);
#endif
}

struct PlaneData {
    Board& board;
    float* inputPlanes;
    float* curIt;
    Stockfish::Color teamSide; 

    PlaneData(Board& board, float* inputPlanes, Stockfish::Color teamSide)
        : board(board), inputPlanes(inputPlanes), curIt(inputPlanes), teamSide(teamSide) {}

    inline void increment_channel() {
        curIt += 64;
    }

    inline void set_plane_to_value(float value) {
        fill_plane_simd(curIt, value);
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

inline void set_plane_pieces_board(PlaneData& p, int boardIdx) { 
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

inline void set_plane_pockets_board(PlaneData& p, int boardIdx) {
    Stockfish::Color first = (boardIdx == 0) ? p.teamSide : ~p.teamSide;
    Stockfish::Color second = ~first;

    for (Stockfish::Color color : {first, second}) {
        for (Stockfish::PieceType piece : {Stockfish::PAWN, Stockfish::KNIGHT, Stockfish::BISHOP, Stockfish::ROOK, Stockfish::QUEEN}) {
            float val = (float)p.board.count_in_hand(boardIdx, color, piece) / 16.0f;
            p.set_plane_to_value(val);
        }
    }
}

inline void set_plane_promoted_pieces_board(PlaneData& p, int boardIdx) {
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

inline void set_plane_ep_square_board(PlaneData& p, int boardIdx) {
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

inline void set_plane_color_info_board(PlaneData& p, int boardIdx) {
    // Python: Board A turn == team_side; Board B turn == not team_side
    Stockfish::Color expectedTurn = (boardIdx == 0) ? p.teamSide : ~p.teamSide;
    float val = (p.board.side_to_move(boardIdx) == expectedTurn) ? 1.0f : 0.0f; 
    p.set_plane_to_value(val);
}

inline void set_plane_castling_rights_board(PlaneData& p, int boardIdx) {
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

void board_to_planes(Board& board, float* inputPlanes, Stockfish::Color teamSide, bool hasTimeAdvantage=false) {
    // Initialize all to 0 using SIMD when available
    // NB_INPUT_VALUES = 64 * 8 * 8 = 4096 floats (exactly 512 AVX2 iterations)
    constexpr size_t totalFloats = 64 * 8 * 8;  // NB_INPUT_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH
#if USE_AVX2
    __m256 zero = _mm256_setzero_ps();
    for (size_t i = 0; i < totalFloats; i += 8) {
        _mm256_storeu_ps(inputPlanes + i, zero);
    }
#else
    std::fill_n(inputPlanes, totalFloats, 0.0f);
#endif
    PlaneData planeData(board, inputPlanes, teamSide);
    
    // Process Board 0 (Channels 0-31)
    set_plane_pieces_board(planeData, 0);           
    set_plane_pockets_board(planeData, 0);          
    set_plane_promoted_pieces_board(planeData, 0);  
    set_plane_ep_square_board(planeData, 0);        
    set_plane_color_info_board(planeData, 0);       
    planeData.set_plane_to_value(1.0f);             // Constant plane
    set_plane_castling_rights_board(planeData, 0);  
    planeData.set_plane_to_value(hasTimeAdvantage ? 1.0f : 0.0f); 
    
    // Process Board 1 (Channels 32-63)
    set_plane_pieces_board(planeData, 1);           
    set_plane_pockets_board(planeData, 1);          
    set_plane_promoted_pieces_board(planeData, 1);  
    set_plane_ep_square_board(planeData, 1);        
    set_plane_color_info_board(planeData, 1);       
    planeData.set_plane_to_value(1.0f);             // Constant plane
    set_plane_castling_rights_board(planeData, 1);  
    planeData.set_plane_to_value(hasTimeAdvantage ? 1.0f : 0.0f);
}