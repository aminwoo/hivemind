use crate::transposition::Bound;
use crate::types::Score;

use super::{defs::SearchRefs, eval::evaluate, Search};
use shakmaty::Position;

impl Search {
    pub fn alpha_beta(
        refs: &mut SearchRefs,
        mut depth: i8,
        mut alpha: i16,
        mut beta: i16,
        null_move: bool,
    ) -> i16 {
        if (refs.search_info.nodes & 2047) == 0
            && refs.search_info.elapsed() > refs.search_params.search_time
        {
            refs.search_info.terminated = true;
            return 0;
        }

        let ply = refs.search_info.ply as usize;
        let is_root = ply == 0;
        let is_frontier = ply == 1;
        let pv_node = beta - alpha > 1;

        let hash = refs.get_hash();

        if !is_root {
            let hit = refs.tt.read(hash.0, ply);
            if let Some(hit) = hit {
                if hit.depth >= depth {
                    refs.search_info.tt_hits += 1;
                    match hit.bound {
                        Bound::Exact => return hit.score,
                        Bound::Lower => alpha = std::cmp::max(alpha, hit.score),
                        Bound::Upper => beta = std::cmp::min(beta, hit.score),
                    }
                    if alpha >= beta {
                        return hit.score;
                    }
                }
            }
        }

        refs.search_info.pv_length[ply] = ply;
        if ply > 0 && refs.three_fold() {
            return 0;
        }
        let is_check = refs.pos.is_check();
        if is_check {
            depth += 1;
        }
        if depth <= 0 {
            return Search::qsearch(refs, alpha, beta);
        }
        refs.search_info.nodes += 1;

        if !is_check && !pv_node {
            let static_eval = evaluate(refs.pos);
            if null_move && depth >= 3 && static_eval >= beta {
                if let Ok(new_pos) = refs.pos.clone().swap_turn() {
                    *refs.pos = new_pos;
                }
                let score = -Search::alpha_beta(refs, depth - 1 - 2, -beta, -beta + 1, false);
                if let Ok(new_pos) = refs.pos.clone().swap_turn() {
                    *refs.pos = new_pos;
                }

                if score >= beta {
                    return beta;
                }
            }
        }

        let mut legal_moves = refs.pos.legal_moves();
        Search::sort_moves(&mut legal_moves, &refs.search_info.pv[ply][ply], refs);

        let mut bound = Bound::Upper;
        let mut fail_low = true;
        let mut pvs = true;
        for m in &legal_moves {
            let prev_pos = refs.pos.clone();
            refs.pos.play_unchecked(m);
            refs.incr_rep();

            refs.search_info.ply += 1;
            refs.search_info.prev_move[ply + 1] = Some(m.clone());

            let mut score;
            if pvs {
                score = -Search::alpha_beta(refs, depth - 1, -beta, -alpha, true);
            } else {
                score = -Search::alpha_beta(refs, depth - 1, -alpha - 1, -alpha, true);
                if score > alpha && beta - alpha > 1 {
                    score = -Search::alpha_beta(refs, depth - 1, -beta, -alpha, true)
                }
            };

            pvs = false;

            refs.decr_rep();
            *refs.pos = prev_pos;
            refs.search_info.ply -= 1;

            if score >= beta {
                refs.tt.write(hash.0, depth, score, Bound::Lower, ply);

                if m.is_capture() {
                    refs.search_info.update_capture_history(
                        m.role(),
                        m.to(),
                        m.capture().unwrap(),
                        1,
                    );
                } else {
                    let ply = refs.search_info.ply as usize;
                    if let Some(first_killer) = &refs.search_info.killer_moves1[ply] {
                        if m != first_killer {
                            refs.search_info.killer_moves2[ply] =
                                refs.search_info.killer_moves1[ply].clone();
                            refs.search_info.killer_moves1[ply] = Some(m.clone());
                        }
                    }

                    refs.search_info.update_quiet_history(
                        m.role(),
                        m.to(),
                        (depth * depth + depth - 1).into(),
                    );
                    if let Some(prev_m) = &refs.search_info.prev_move[ply] {
                        refs.search_info.counter_moves[prev_m.from().unwrap() as usize]
                            [prev_m.to() as usize] = Some(m.clone());
                    }
                }
                return beta;
            } else if m.is_capture() {
                refs.search_info
                    .update_capture_history(m.role(), m.to(), m.capture().unwrap(), -1);
            }

            if score > alpha {
                fail_low = false;
                alpha = score;
                bound = Bound::Exact;

                refs.search_info.pv[ply].fill(None);
                refs.search_info.pv[ply][ply] = Some(m.clone());
                for next_ply in ply + 1..refs.search_info.pv_length[ply + 1] {
                    refs.search_info.pv[ply][next_ply] =
                        refs.search_info.pv[ply + 1][next_ply].clone();
                }
                refs.search_info.pv_length[ply] = refs.search_info.pv_length[ply + 1];
            } else if is_root && fail_low {
                return -Score::INFINITY;
            }
        }

        if legal_moves.is_empty() {
            if is_check {
                return -Score::INFINITY + (refs.search_info.ply as i16);
            } else {
                return 0;
            }
        }

        refs.tt.write(hash.0, depth, alpha, bound, ply);
        alpha
    }
}
