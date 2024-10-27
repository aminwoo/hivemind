use super::defs::SearchRefs;
use super::{Move, Search};
use shakmaty::{MoveList, Position};

const TTMOVE_SORT_VALUE: i16 = 60;
const MVV_LVA_OFFSET: i16 = 5000;
const KILLER_VALUE: i16 = 10;
const PROMOTION_VALUE: i16 = 56;

pub const MVV_LVA: [[i16; 7]; 7] = [
    [0, 0, 0, 0, 0, 0, 0],       // victim None, attacker None, P, N, B, R, Q, K
    [0, 15, 14, 13, 12, 11, 10], // victim P, attacker None, P, N, B, R, Q, K
    [0, 25, 24, 23, 22, 21, 20], // victim N, attacker None, P, N, B, R, Q, K
    [0, 35, 34, 33, 32, 31, 30], // victim B, attacker None, P, N, B, R, Q, K
    [0, 45, 44, 43, 42, 41, 40], // victim R, attacker None, P, N, B, R, Q, K
    [0, 55, 54, 53, 52, 51, 50], // victim Q, attacker None, P, N, B, R, Q, K
    [0, 0, 0, 0, 0, 0, 0],       // victim K, attacker None, P, N, B, R, Q, K
];

impl Search {
    pub fn score_moves(
        moves: &mut MoveList,
        tt_move: &Option<Move>,
        refs: &SearchRefs,
    ) -> [i16; 128] {
        let mut scores: [i16; 128] = [0; 128];
        for (i, mv) in moves.iter().enumerate() {
            if let Some(m) = &tt_move {
                if mv == m {
                    scores[i] = MVV_LVA_OFFSET + TTMOVE_SORT_VALUE;
                    continue;
                }
            }

            if mv.is_promotion() {
                scores[i] = MVV_LVA_OFFSET + PROMOTION_VALUE;
            } else if mv.is_capture() {
                let piece = mv.role() as usize;
                let captured = match mv.capture() {
                    Some(role) => role as usize,
                    None => 0,
                };
                scores[i] = MVV_LVA_OFFSET + MVV_LVA[captured][piece];
            } else {
                let ply = refs.search_info.ply as usize;
                let mut is_killer = false;
                if let Some(first_killer) = &refs.search_info.killer_moves1[ply] {
                    if mv == first_killer {
                        scores[i] = MVV_LVA_OFFSET - KILLER_VALUE;
                        is_killer = true;
                    }
                }
                if let Some(second_killer) = &refs.search_info.killer_moves2[ply] {
                    if mv == second_killer {
                        scores[i] = MVV_LVA_OFFSET - KILLER_VALUE;
                        is_killer = true;
                    }
                }
                if !is_killer {
                    scores[i] = refs.search_info.history[mv.role() as usize][mv.to() as usize];
                }
            }
        }
        scores
    }

    pub fn pick_move(moves: &mut MoveList, start_index: usize, scores: [i16; 128]) {
        for i in (start_index + 1)..moves.len() {
            if scores[i] > scores[start_index] {
                moves.swap(start_index, i);
            }
        }
    }
}
