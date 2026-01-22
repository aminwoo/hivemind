#include "planes.h"
#include <algorithm>

// Stockfish bitboards are typically 64-bit integers where bit 0 is A1.
// This function maps the bitboard to the current plane iterator.
inline void set_bits_from_bitmap(Stockfish::Bitboard bb, float *curIt) {
    for (int i = 0; i < 64; i++) {
        if (bb & (1ULL << i)) {
            curIt[i] = 1.0f;
        }
    }
}

struct PlaneData {
    Board& board;
    float* inputPlanes;
    float* curIt;
    Stockfish::Color us; 

    PlaneData(Board& board, float* inputPlanes, Stockfish::Color us)
        : board(board), inputPlanes(inputPlanes), curIt(inputPlanes), us(us) {}

    inline void increment_channel() {
        curIt += 64;
    }

    inline void set_plane_to_value(float value) {
        std::fill_n(curIt, 64, value);
        increment_channel();
    }

    inline void set_plane_to_bitboard(Stockfish::Bitboard bb) {
        set_bits_from_bitmap(bb, curIt);
        increment_channel();
    }

    // Helper to determine if the specific board needs perspective flipping
    inline bool needs_flipping(int boardIdx) {
        if (boardIdx == 0) return us == Stockfish::BLACK;
        return us == Stockfish::WHITE;
    }
};

inline void set_plane_pieces_board(PlaneData& p, int boardIdx) { 
    // Python logic: Board A uses (us, opponent), Board B uses (opponent, us)
    Stockfish::Color first = (boardIdx == 0) ? p.us : ~p.us;
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
    Stockfish::Color first = (boardIdx == 0) ? p.us : ~p.us;
    Stockfish::Color second = ~first;

    for (Stockfish::Color color : {first, second}) {
        for (Stockfish::PieceType piece : {Stockfish::PAWN, Stockfish::KNIGHT, Stockfish::BISHOP, Stockfish::ROOK, Stockfish::QUEEN}) {
            float val = (float)p.board.count_in_hand(boardIdx, color, piece) / 16.0f;
            p.set_plane_to_value(val);
        }
    }
}

inline void set_plane_promoted_pieces_board(PlaneData& p, int boardIdx) {
    Stockfish::Color first = (boardIdx == 0) ? p.us : ~p.us;
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
    Stockfish::Color expectedTurn = (boardIdx == 0) ? p.us : ~p.us;
    float val = (p.board.side_to_move(boardIdx) == expectedTurn) ? 1.0f : 0.0f; 
    p.set_plane_to_value(val);
}

inline void set_plane_castling_rights_board(PlaneData& p, int boardIdx) {
    // Note: Castling rights are usually evaluated on the original board state
    if (boardIdx == 0) {
        p.set_plane_to_value(p.board.can_castle(0, p.us == Stockfish::WHITE ? Stockfish::WHITE_OO : Stockfish::BLACK_OO));
        p.set_plane_to_value(p.board.can_castle(0, p.us == Stockfish::WHITE ? Stockfish::WHITE_OOO : Stockfish::BLACK_OOO));
        p.set_plane_to_value(p.board.can_castle(0, p.us == Stockfish::WHITE ? Stockfish::BLACK_OO : Stockfish::WHITE_OO));
        p.set_plane_to_value(p.board.can_castle(0, p.us == Stockfish::WHITE ? Stockfish::BLACK_OOO : Stockfish::WHITE_OOO));
    } else {
        // Board B logic: Kingside(not team_side), Queenside(not team_side), Kingside(team_side), Queenside(team_side)
        p.set_plane_to_value(p.board.can_castle(1, p.us == Stockfish::WHITE ? Stockfish::BLACK_OO : Stockfish::WHITE_OO));
        p.set_plane_to_value(p.board.can_castle(1, p.us == Stockfish::WHITE ? Stockfish::BLACK_OOO : Stockfish::WHITE_OOO));
        p.set_plane_to_value(p.board.can_castle(1, p.us == Stockfish::WHITE ? Stockfish::WHITE_OO : Stockfish::BLACK_OO));
        p.set_plane_to_value(p.board.can_castle(1, p.us == Stockfish::WHITE ? Stockfish::WHITE_OOO : Stockfish::BLACK_OOO));
    }
}

void board_to_planes(Board& board, float* inputPlanes, Stockfish::Color us, bool has_time_advantage=false) {
    // Initialize all to 0
    std::fill_n(inputPlanes, NB_INPUT_VALUES(), 0.0f); 
    PlaneData planeData(board, inputPlanes, us);
    
    // Process Board 0 (Channels 0-31)
    set_plane_pieces_board(planeData, 0);           
    set_plane_pockets_board(planeData, 0);          
    set_plane_promoted_pieces_board(planeData, 0);  
    set_plane_ep_square_board(planeData, 0);        
    set_plane_color_info_board(planeData, 0);       
    planeData.set_plane_to_value(1.0f);             // Constant plane
    set_plane_castling_rights_board(planeData, 0);  
    planeData.set_plane_to_value(has_time_advantage ? 1.0f : 0.0f); 
    
    // Process Board 1 (Channels 32-63)
    set_plane_pieces_board(planeData, 1);           
    set_plane_pockets_board(planeData, 1);          
    set_plane_promoted_pieces_board(planeData, 1);  
    set_plane_ep_square_board(planeData, 1);        
    set_plane_color_info_board(planeData, 1);       
    planeData.set_plane_to_value(1.0f);             // Constant plane
    set_plane_castling_rights_board(planeData, 1);  
    planeData.set_plane_to_value(has_time_advantage ? 1.0f : 0.0f);
}