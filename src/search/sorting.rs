use super::defs::SearchRefs;
use super::{Move, Search};
use crate::types::parameters::*;
use shakmaty::{Bitboard, MoveList, Position};

const BAD_CAPTURE: i32 = -200_000_000;
const GOOD_CAPTURE: i32 = 200_000_000;
const KILLER_BONUS: i32 = 100_000_000;
const HASH_MOVE: i32 = 300_000_000;

pub const MVV_LVA: [[i32; 7]; 7] = [
    [0, 0, 0, 0, 0, 0, 0],       // victim None, attacker None, P, N, B, R, Q, K
    [0, 6, 5, 4, 3, 2, 1],       // victim P, attacker None, P, N, B, R, Q, K
    [0, 12, 11, 10, 9, 8, 7],    // victim N, attacker None, P, N, B, R, Q, K
    [0, 18, 17, 16, 15, 14, 13], // victim B, attacker None, P, N, B, R, Q, K
    [0, 24, 23, 22, 21, 20, 19], // victim R, attacker None, P, N, B, R, Q, K
    [0, 30, 29, 28, 27, 26, 25], // victim Q, attacker None, P, N, B, R, Q, K
    [0, 0, 0, 0, 0, 0, 0],       // victim K, attacker None, P, N, B, R, Q, K
];

pub const SEE_VALUE: [i32; 7] = [0, 100, 325, 325, 500, 1000, 30000];

pub fn see(refs: &SearchRefs, m: &Move) -> i32 {
    let pos = &refs.pos;
    let board = pos.board();

    let mut gain: [i32; 32] = [0; 32];
    let mut depth = 0;

    let mut from_set = Bitboard::from(m.from().unwrap());
    let mut occupied = board.occupied();

    let mut target = 0;
    let mut attacker = m.role() as usize;
    if let Some(role) = m.capture() {
        target = role as usize;
    }
    gain[depth] = SEE_VALUE[target];

    let mut attackers;
    loop {
        depth += 1;
        gain[depth] = SEE_VALUE[attacker] - gain[depth - 1];
        occupied ^= from_set;

        if depth & 1 == 1 {
            attackers = board.attacks_to(m.to(), pos.turn().other(), occupied) & occupied;
        } else {
            attackers = board.attacks_to(m.to(), pos.turn(), occupied) & occupied;
        }

        from_set = attackers & board.pawns();
        if from_set.any() {
            from_set = from_set.isolate_first();
            attacker = 1;
            continue;
        }
        from_set = attackers & board.knights();
        if from_set.any() {
            from_set = from_set.isolate_first();
            attacker = 2;
            continue;
        }
        from_set = attackers & board.bishops();
        if from_set.any() {
            from_set = from_set.isolate_first();
            attacker = 3;
            continue;
        }
        from_set = attackers & board.rooks();
        if from_set.any() {
            from_set = from_set.isolate_first();
            attacker = 4;
            continue;
        }
        from_set = attackers & board.queens();
        if from_set.any() {
            from_set = from_set.isolate_first();
            attacker = 5;
            continue;
        }
        break;
    }

    depth -= 1;
    while depth > 0 {
        gain[depth - 1] = -std::cmp::max(-gain[depth - 1], gain[depth]);
        depth -= 1;
    }
    gain[0]
}

impl Search {
    pub fn sort_moves(
        moves: &mut MoveList,
        pv_move: &Option<Move>,
        tt_move: &Option<Move>,
        refs: &SearchRefs,
    ) {
        moves.sort_by_key(|m| {
            if let Some(mv) = &pv_move {
                if mv == m {}
            }
            if let Some(mv) = &tt_move {
                if mv == m {
                    return HASH_MOVE;
                }
            }

            if m.is_capture() {
                let piece = m.role() as usize;
                let captured = match m.capture() {
                    Some(role) => role as usize,
                    None => 0,
                };
                let see_value = see(refs, m);
                let history = refs
                    .search_info
                    .history
                    .get_capture(refs.pos.turn(), m.clone())
                    .expect("Not capture");
                let mvv = MVV_LVA[captured][piece];
                if see_value < 0 {
                    return BAD_CAPTURE + see_value + history + mvv;
                }
                return GOOD_CAPTURE + see_value + history + mvv;
            }
            let ply = refs.search_info.ply as usize;
            if let Some(killer) = &refs.search_info.killers[ply] {
                if m == killer {
                    return KILLER_BONUS;
                }
            }
            ordering_main()
                * refs
                    .search_info
                    .history
                    .get_main(refs.pos.turn(), m.clone())
                    .expect("Non drop")
        });
        moves.reverse();
    }
}
