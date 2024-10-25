use shakmaty::{Color, Position};

type Psqt = [i16; 64];

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
    870, 880, 890, 890, 890, 890, 880, 870,
    880, 890, 895, 895, 895, 895, 890, 880,
    890, 895, 910, 910, 910, 910, 895, 890,
    890, 895, 910, 920, 920, 910, 895, 890,
    890, 895, 910, 920, 920, 910, 895, 890,
    890, 895, 895, 895, 895, 895, 895, 890,
    880, 890, 895, 895, 895, 895, 890, 880,
    870, 880, 890, 890, 890, 890, 880, 870 
];

#[rustfmt::skip]
const ROOK_MG: Psqt = [
   500, 500, 500, 500, 500, 500, 500, 500,
   515, 515, 515, 520, 520, 515, 515, 515,
   500, 500, 500, 500, 500, 500, 500, 500,
   500, 500, 500, 500, 500, 500, 500, 500,
   500, 500, 500, 500, 500, 500, 500, 500,
   500, 500, 500, 500, 500, 500, 500, 500,
   500, 500, 500, 500, 500, 500, 500, 500,
   500, 500, 500, 510, 510, 510, 500, 500
];

#[rustfmt::skip]
const BISHOP_MG: Psqt = [
    300, 320, 320, 320, 320, 320, 320, 300,
    305, 320, 320, 320, 320, 320, 320, 305,
    310, 320, 320, 325, 325, 320, 320, 310,
    310, 330, 330, 350, 350, 330, 330, 310,
    325, 325, 330, 345, 345, 330, 325, 325,
    325, 325, 325, 330, 330, 325, 325, 325,
    310, 325, 325, 330, 330, 325, 325, 310,
    300, 310, 310, 310, 310, 310, 310, 300
];

#[rustfmt::skip]
const KNIGHT_MG: Psqt = [
    290, 300, 300, 300, 300, 300, 300, 290,
    300, 305, 305, 305, 305, 305, 305, 300,
    300, 305, 325, 325, 325, 325, 305, 300,
    300, 305, 325, 325, 325, 325, 305, 300,
    300, 305, 325, 325, 325, 325, 305, 300,
    300, 305, 320, 325, 325, 325, 305, 300,
    300, 305, 305, 305, 305, 305, 305, 300,
    290, 310, 300, 300, 300, 300, 310, 290
];

#[rustfmt::skip]
const PAWN_MG: Psqt = [
    100, 100, 100, 100, 100, 100, 100, 100,
    160, 160, 160, 160, 170, 160, 160, 160,
    140, 140, 140, 150, 160, 140, 140, 140,
    120, 120, 120, 140, 150, 120, 120, 120,
    105, 105, 115, 130, 140, 110, 105, 105,
    105, 105, 110, 120, 130, 105, 105, 105,
    105, 105, 105,  70,  70, 105, 105, 105,
    100, 100, 100, 100, 100, 100, 100, 100
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

pub fn evaluate<T: Position>(pos: &T) -> f32 {
    let mut score: f32;
    let mut w_psqt = 0.0;
    let mut b_psqt = 0.0;

    let bb_white = pos.board().white();
    let bb_black = pos.board().black();

    for square in bb_white {
        let piece_result = pos.board().piece_at(square);
        if let Some(piece) = piece_result {
            w_psqt += PSQT_MG[piece.role as usize][FLIP[square as usize]] as f32;
        }
    }
    for square in bb_black {
        let piece_result = pos.board().piece_at(square);
        if let Some(piece) = piece_result {
            b_psqt += PSQT_MG[piece.role as usize][square as usize] as f32;
        }
    }

    score = w_psqt - b_psqt;

    /*score += (pos.board().white() & pos.board().pawns()).count() as f32 * 1.0;
    score += (pos.board().white() & pos.board().knights()).count() as f32 * 2.0;
    score += (pos.board().white() & pos.board().bishops()).count() as f32 * 1.7;
    score += (pos.board().white() & pos.board().rooks()).count() as f32 * 3.2;
    score += (pos.board().white() & pos.board().queens()).count() as f32 * 4.6;

    score -= (pos.board().black() & pos.board().pawns()).count() as f32 * 1.0;
    score -= (pos.board().black() & pos.board().knights()).count() as f32 * 2.0;
    score -= (pos.board().black() & pos.board().bishops()).count() as f32 * 1.7;
    score -= (pos.board().black() & pos.board().rooks()).count() as f32 * 3.2;
    score -= (pos.board().black() & pos.board().queens()).count() as f32 * 4.6;*/

    /*let pockets_result = pos.pockets();
    if let Some(pockets) = pockets_result {
        score += *pockets.get(Color::White).get(Role::Queen) as f32 * 0.7;
        score += *pockets.get(Color::White).get(Role::Rook) as f32 * 1.5;
        score += *pockets.get(Color::White).get(Role::Bishop) as f32 * 1.4;
        score += *pockets.get(Color::White).get(Role::Knight) as f32 * 2.4;
        score += *pockets.get(Color::White).get(Role::Pawn) as f32 * 3.7;

        score -= *pockets.get(Color::Black).get(Role::Queen) as f32 * 0.7;
        score -= *pockets.get(Color::Black).get(Role::Rook) as f32 * 1.5;
        score -= *pockets.get(Color::Black).get(Role::Bishop) as f32 * 1.4;
        score -= *pockets.get(Color::Black).get(Role::Knight) as f32 * 2.4;
        score -= *pockets.get(Color::Black).get(Role::Pawn) as f32 * 3.7;
    }*/

    if pos.turn() == Color::Black {
        score *= -1.0;
    }

    score
}
