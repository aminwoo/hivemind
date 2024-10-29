use crate::engine::transposition::{HashFlag, SearchData};

use super::{defs::SearchRefs, Search};
use shakmaty::{
    zobrist::{Zobrist64, ZobristHash},
    EnPassantMode, Position,
};

impl Search {
    pub fn alpha_beta(refs: &mut SearchRefs, mut depth: i8, mut alpha: i16, mut beta: i16) -> i16 {
        if refs.search_info.elapsed() > refs.search_params.search_time {
            refs.search_info.terminated = true;
            return 0;
        }

        let ply = refs.search_info.ply as usize;
        let is_root = ply == 0;
        let is_frontier = ply == 1;
        refs.search_info.pv_length[ply] = ply;

        let is_check = refs.pos.is_check();
        if is_check {
            depth += 1;
        }

        if depth <= 0 {
            return Search::qsearch(refs, alpha, beta);
        }

        refs.search_info.nodes += 1;

        let mut tt_value: Option<i16> = None;
        let mut tt_flag: HashFlag = HashFlag::Nothing;
        if refs.tt_enabled {
            if let Some(data) = refs
                .tt
                .probe(refs.pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal))
            {
                (tt_value, tt_flag) = data.get(depth, ply, alpha, beta);
            }
        }

        if let Some(v) = tt_value {
            if !is_root {
                match tt_flag {
                    HashFlag::Exact => return v,
                    HashFlag::Beta => alpha = std::cmp::max(alpha, v),
                    HashFlag::Alpha => beta = std::cmp::min(beta, v),
                    _ => (),
                }
                refs.search_info.tt_hits += 1;
            }
        }

        let mut hash_flag = HashFlag::Alpha;

        let mut legal_moves = refs.pos.legal_moves();
        Search::sort_moves(&mut legal_moves, &refs.search_info.pv[ply][ply], refs);

        let mut fail_low = true;
        let mut pvs = true;
        for m in &legal_moves {
            let prev_pos = refs.pos.clone();
            refs.pos.play_unchecked(m);
            refs.search_info.ply += 1;

            let mut score;
            if pvs {
                score = -Search::alpha_beta(refs, depth - 1, -beta, -alpha);
            } else {
                score = -Search::alpha_beta(refs, depth - 1, -alpha - 1, -alpha);
                if score > alpha && beta - alpha > 1 {
                    score = -Search::alpha_beta(refs, depth - 1, -beta, -alpha)
                }
            };

            pvs = false;

            *refs.pos = prev_pos;
            refs.search_info.ply -= 1;

            if score >= beta {
                refs.tt.insert(
                    refs.pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal),
                    SearchData::create(depth, ply, HashFlag::Beta, beta),
                );

                if !m.is_capture() {
                    let ply = refs.search_info.ply as usize;
                    if let Some(first_killer) = &refs.search_info.killer_moves1[ply] {
                        if m != first_killer {
                            refs.search_info.killer_moves2[ply] =
                                refs.search_info.killer_moves1[ply].clone();
                            refs.search_info.killer_moves1[ply] = Some(m.clone());
                        }
                    }

                    refs.search_info
                        .update_history(m.role(), m.to(), (depth * depth).into());
                    refs.search_info.counter_moves[m.from().unwrap() as usize][m.to() as usize] =
                        Some(m.clone());
                }
                return beta;
            }

            if score > alpha {
                fail_low = false;
                alpha = score;
                hash_flag = HashFlag::Exact;

                refs.search_info.pv[ply].fill(None);
                refs.search_info.pv[ply][ply] = Some(m.clone());
                for next_ply in ply + 1..refs.search_info.pv_length[ply + 1] {
                    refs.search_info.pv[ply][next_ply] =
                        refs.search_info.pv[ply + 1][next_ply].clone();
                }
                refs.search_info.pv_length[ply] = refs.search_info.pv_length[ply + 1];
            } else if is_root && fail_low {
                return -10000;
            }
        }

        if legal_moves.is_empty() {
            if is_check {
                return -9690 + (refs.search_info.ply as i16);
            } else {
                return 0;
            }
        }
        refs.tt.insert(
            refs.pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal),
            SearchData::create(depth, ply, hash_flag, alpha),
        );
        alpha
    }
}
