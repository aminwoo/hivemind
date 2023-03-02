#include "planes.h"

inline void set_bits_from_bitmap(Stockfish::Bitboard a, Stockfish::Bitboard b, float *curIt) {
    // set the individual bits for the pieces
    // https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/
    for (int i = 0; i < BOARD_HEIGHT; i++) {
        for (int j = 0; j < BOARD_WIDTH; j++) {
            if (a & 0x1) {
                *curIt = 1;
            }
            a >>= 1;
            ++curIt;
        }
        for (int j = 0; j < BOARD_WIDTH; j++) {
            if (b & 0x1) {
                *curIt = 1;
            }
            b >>= 1;
            ++curIt;
        }
    }
}

struct PlaneData {
    Bugboard& board;
    float* inputPlanes;
    float* curIt;
    Stockfish::Color us; 
    PlaneData(Bugboard& board, float* inputPlanes, Stockfish::Color us): board(board), inputPlanes(inputPlanes), curIt(inputPlanes), us(us) {}

    inline void increment_channel() {
        curIt += 128;
    }

    inline size_t current_channel() {
        return (curIt - inputPlanes) / 128;
    }

    inline void set_plane_to_value(float value) {
        std::fill_n(curIt, 128, value);
        increment_channel();
    }

    inline void set_plane_to_values(float a, float b) {
        for (int i = 0; i < BOARD_HEIGHT; i++) {
            for (int j = 0; j < BOARD_WIDTH; j++) {
                *curIt = a;
                ++curIt;
            }
            for (int j = 0; j < BOARD_WIDTH; j++) {
                *curIt = b;
                ++curIt;
            }
        }
    }
    inline void set_plane_to_bitboard(Stockfish::Bitboard a, Stockfish::Bitboard b) {
        set_bits_from_bitmap(a, b, curIt);
        increment_channel();
    }
};

inline void set_plane_pieces(PlaneData& p) { 
    for (Stockfish::Color color : {p.us, ~p.us}) {
        for (Stockfish::PieceType piece: {Stockfish::PAWN, Stockfish::KNIGHT, Stockfish::BISHOP, Stockfish::ROOK, Stockfish::QUEEN, Stockfish::KING}) {
            Stockfish::Bitboard a = p.board.pieces(0, color, piece);
            Stockfish::Bitboard b = p.board.pieces(1, ~color, piece);
            if (p.us == Stockfish::BLACK) {
                a = flip_vertical(a);
            } 
            else {
                b = flip_vertical(b);
            }
            p.set_plane_to_bitboard(a, b);
        }
    }
}

inline void set_plane_pockets(PlaneData& p) {
    for (Stockfish::Color color : {p.us, ~p.us}) {
        for (Stockfish::PieceType piece: {Stockfish::PAWN, Stockfish::KNIGHT, Stockfish::BISHOP, Stockfish::ROOK, Stockfish::QUEEN}) {
            float a = p.board.count_in_hand(0, color, piece) / 16.0;
            float b = p.board.count_in_hand(1, ~color, piece) / 16.0;
            p.set_plane_to_values(a, b);
        }
    }
}

inline void set_plane_promoted_pieces(PlaneData& p) {
    for (Stockfish::Color color : {p.us, ~p.us}) {
        Stockfish::Bitboard a = p.board.promoted_pieces(0) & p.board.pieces(0, color); 
        Stockfish::Bitboard b = p.board.promoted_pieces(1) & p.board.pieces(1, ~color); 
        if (p.us == Stockfish::BLACK) {
            a = flip_vertical(a);
        } 
        else {
            b = flip_vertical(b);
        }
        p.set_plane_to_bitboard(a, b);
    }
}

inline void set_plane_ep_square(PlaneData& p) {
    Stockfish::Bitboard a = 0, b = 0; 
    if (p.board.ep_square(0) != Stockfish::SQ_NONE) {
        a = 1LL << p.board.ep_square(0);
    }
    if (p.board.ep_square(1) != Stockfish::SQ_NONE) {
        b = 1LL << p.board.ep_square(1); 
    }

    if (p.us == Stockfish::BLACK) {
        a = flip_vertical(a);
    } 
    else {
        b = flip_vertical(b);
    }
    p.set_plane_to_bitboard(a, b);
}

inline void set_plane_color_info(PlaneData& p) {
    float a = p.board.side_to_move(0) == p.us; 
    float b = p.board.side_to_move(1) == ~p.us; 
    p.set_plane_to_value(a);
    p.set_plane_to_value(b);
}

inline void set_plane_castling_rights(PlaneData& p) {
    int a, b; 
    if (p.us == Stockfish::WHITE) {
        a = p.board.can_castle(0, Stockfish::WHITE_OO);
        b = p.board.can_castle(1, Stockfish::BLACK_OO);
    }
    else {
        a = p.board.can_castle(0, Stockfish::BLACK_OO);
        b = p.board.can_castle(1, Stockfish::WHITE_OO);
    }
    p.set_plane_to_values(a, b);

    if (p.us == Stockfish::WHITE) {
        a = p.board.can_castle(0, Stockfish::WHITE_OOO);
        b = p.board.can_castle(1, Stockfish::BLACK_OOO);
    }
    else {
        a = p.board.can_castle(0, Stockfish::BLACK_OOO);
        b = p.board.can_castle(1, Stockfish::WHITE_OOO);
    }
    p.set_plane_to_values(a, b);

    if (p.us == Stockfish::WHITE) {
        a = p.board.can_castle(0, Stockfish::BLACK_OO);
        b = p.board.can_castle(1, Stockfish::WHITE_OO);
    }
    else {
        a = p.board.can_castle(0, Stockfish::WHITE_OO);
        b = p.board.can_castle(1, Stockfish::BLACK_OO);
    }
    p.set_plane_to_values(a, b);

    if (p.us == Stockfish::WHITE) {
        a = p.board.can_castle(0, Stockfish::BLACK_OOO);
        b = p.board.can_castle(1, Stockfish::WHITE_OOO);
    }
    else {
        a = p.board.can_castle(0, Stockfish::WHITE_OOO);
        b = p.board.can_castle(1, Stockfish::BLACK_OOO);
    }
    p.set_plane_to_values(a, b);
}

inline void set_time_difference(PlaneData& p) {
    Clock clock = p.board.get_clock(); 
    
    if (p.us == Stockfish::WHITE) {
        p.set_plane_to_value(0.5 + (clock.get_time(0, Stockfish::WHITE) - clock.get_time(1, Stockfish::WHITE)) / 1800.0);
    }
    else {
        p.set_plane_to_value(0.5 + (clock.get_time(0, Stockfish::BLACK) - clock.get_time(1, Stockfish::BLACK)) / 1800.0);
    }
}

void board_to_planes(Bugboard& board, float* inputPlanes, Stockfish::Color us) {
    std::fill_n(inputPlanes, NUM_BUGHOUSE_VALUES(), 0.0f);
    PlaneData planeData(board, inputPlanes, us);
    set_plane_pieces(planeData);
    set_plane_pockets(planeData);
    set_plane_promoted_pieces(planeData);
    set_plane_ep_square(planeData);
    set_plane_color_info(planeData);
    set_plane_castling_rights(planeData);
    set_time_difference(planeData); 
    //set_scramble(planeData); 
}











