use super::defs::SearchRefs;
use super::{Move, Search};
use shakmaty::{Bitboard, Color, MoveList, Position, Role};

const CAPTURE_BONUS: i16 = 10000;
const PROMOTION_BONUS: i16 = 20000;

pub const MVV_LVA: [[i16; 7]; 7] = [
    [0, 0, 0, 0, 0, 0, 0],       // victim None, attacker None, P, N, B, R, Q, K
    [0, 6, 5, 4, 3, 2, 1],       // victim P, attacker None, P, N, B, R, Q, K
    [0, 12, 11, 10, 9, 8, 7],    // victim N, attacker None, P, N, B, R, Q, K
    [0, 18, 17, 16, 15, 14, 13], // victim B, attacker None, P, N, B, R, Q, K
    [0, 24, 23, 22, 21, 20, 19], // victim R, attacker None, P, N, B, R, Q, K
    [0, 30, 29, 28, 27, 26, 25], // victim Q, attacker None, P, N, B, R, Q, K
    [0, 0, 0, 0, 0, 0, 0],       // victim K, attacker None, P, N, B, R, Q, K
];

pub const SEE_VALUE: [i16; 7] = [0, 100, 325, 325, 500, 1000, 30000];

pub fn see(refs: &SearchRefs, m: &Move) -> i16 {
    let pos = &refs.pos;
    let board = pos.board();

    let mut gain: [i16; 32] = [0; 32];
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
    pub fn sort_moves(moves: &mut MoveList, tt_move: &Option<Move>, refs: &SearchRefs) {
        moves.sort_by_key(|m| {
            if let Some(mv) = &tt_move {
                if mv == m {
                    return i16::MAX;
                }
            }
            if m.is_promotion() {
                PROMOTION_BONUS
            } else if m.is_capture() {
                let piece = m.role() as usize;
                let captured = match m.capture() {
                    Some(role) => role as usize,
                    None => 0,
                };
                let see_value = see(refs, m);
                let capture_score = refs.search_info.get_capture_score(m);
                let mvv = MVV_LVA[captured][piece];
                if see_value < 0 {
                    return see_value + mvv;
                }
                return CAPTURE_BONUS + see_value + capture_score;
            } else {
                let ply = refs.search_info.ply as usize;
                if let Some(first_killer) = &refs.search_info.killer_moves1[ply] {
                    if m == first_killer {
                        return 6000;
                    }
                }
                if let Some(second_killer) = &refs.search_info.killer_moves2[ply] {
                    if m == second_killer {
                        return 5000;
                    }
                }
                if let Some(prev_m) = &refs.search_info.prev_move[ply] {
                    if let Some(counter) = &refs.search_info.counter_moves
                        [prev_m.from().unwrap() as usize][prev_m.to() as usize]
                    {
                        if m == counter {
                            /*println!("{:?}", refs.pos.board());
                            println!("{:?}", prev_m);
                            println!("{:?}", m);
                            println!();*/
                            return 3000;
                        }
                    }
                }
                refs.search_info.get_quiet_score(m)
            }
        });
        moves.reverse();
    }
}
