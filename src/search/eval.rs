use shakmaty::{Bitboard, Chess, Color, Position};

type Psqt = [i16; 64];

const MOBILITY_BONUS: [[i16; 28]; 7] = [
    [0; 28],
    [0; 28],
    [
        -75, -56, -9, -2, 6, 15, 22, 30, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0,
    ], // Knights
    [
        -48, -21, 16, 26, 37, 51, 54, 63, 65, 71, 79, 81, 92, 97, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0,
    ], // Bishops
    [
        -56, -25, -11, -5, -4, -1, 8, 14, 21, 23, 31, 32, 43, 49, 59, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0,
    ], // Rooks
    [
        -40, -25, 2, 4, 14, 24, 25, 40, 43, 47, 54, 56, 60, 70, 72, 73, 75, 77, 85, 94, 99, 108,
        112, 113, 118, 119, 123, 128,
    ], // Queens
    [0; 28],
];
const PIECE_VALUES: [i16; 7] = [0, 100, 320, 330, 500, 900, 20000];

#[rustfmt::skip]
const KING_MG: Psqt = [
    0, 0,  0,   0,   0, 0,  0, 0,
    0, 0,  0,   0,   0, 0,  0, 0,
    0, 0,  0,   0,   0, 0,  0, 0,
    0, 0,  0,  20,  20, 0,  0, 0,
    0, 0,  0,  20,  20, 0,  0, 0,
    0, 0,  0,   0,   0, 0,  0, 0,
    0, 0,  0, -10, -10, 0,  0, 0,
    0, 0, 20, -10, -10, 0, 20, 0,
];

#[rustfmt::skip]
const QUEEN_MG: Psqt = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
];

#[rustfmt::skip]
const ROOK_MG: Psqt = [
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0
];

#[rustfmt::skip]
const BISHOP_MG: Psqt = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];

#[rustfmt::skip]
const KNIGHT_MG: Psqt = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];

#[rustfmt::skip]
const PAWN_MG: Psqt = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
];

pub const PSQT_MG: [Psqt; 7] = [
    [0; 64], PAWN_MG, KNIGHT_MG, BISHOP_MG, ROOK_MG, QUEEN_MG, KING_MG,
];

#[allow(dead_code)]
#[rustfmt::skip]
pub const FLIP: [usize; 64] = [
    56, 57, 58, 59, 60, 61, 62, 63,
    48, 49, 50, 51, 52, 53, 54, 55,
    40, 41, 42, 43, 44, 45, 46, 47,
    32, 33, 34, 35, 36, 37, 38, 39,
    24, 25, 26, 27, 28, 29, 30, 31,
    16, 17, 18, 19, 20, 21, 22, 23,
     8,  9, 10, 11, 12, 13, 14, 15,
     0,  1,  2,  3,  4,  5,  6,  7,
];

pub fn calculate_mobility(pos: &Chess, occupied: Bitboard) -> i16 {
    let mut mobility_score = 0;
    for square in occupied {
        let mobility_count = pos.board().attacks_from(square).count();
        let piece = pos.board().piece_at(square);
        if let Some(piece) = piece {
            let role = piece.role as usize;
            mobility_score += MOBILITY_BONUS[role][mobility_count];
        }
    }
    mobility_score
}

pub fn calculate_king_safety(pos: &Chess, occupied: Bitboard) -> i16 {
    0
}

pub fn evaluate(pos: &Chess) -> i16 {
    let mut score: i16;
    let mut w_psqt = 0;
    let mut b_psqt = 0;

    let bb_white = pos.board().white();
    let bb_black = pos.board().black();
    let w_mobility = calculate_mobility(pos, bb_white);
    let b_mobility = calculate_mobility(pos, bb_black);
    let w_king = calculate_king_safety(pos, bb_white);
    let b_king = calculate_king_safety(pos, bb_black);

    for square in bb_white {
        let piece_result = pos.board().piece_at(square);
        if let Some(piece) = piece_result {
            let role = piece.role as usize;
            w_psqt += PIECE_VALUES[role] + PSQT_MG[role][FLIP[square as usize]];
        }
    }
    for square in bb_black {
        let piece_result = pos.board().piece_at(square);
        if let Some(piece) = piece_result {
            let role = piece.role as usize;
            b_psqt += PIECE_VALUES[role] + PSQT_MG[role][square as usize];
        }
    }

    score = w_psqt + w_mobility + w_king - b_psqt - b_mobility - b_king;

    if pos.turn() == Color::Black {
        score *= -1;
    }

    score
}
