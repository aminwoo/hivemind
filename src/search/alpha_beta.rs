use crate::transposition::Bound;
use crate::types::Score;

use super::{defs::SearchRefs, eval::evaluate, Search};
use shakmaty::{Move, Position};

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

        let mut tt_move: Option<Move> = None;
        if !is_root {
            let hit = refs.tt.read(hash.0, ply);
            if let Some(hit) = hit {
                if hit.depth >= depth {
                    match hit.bound {
                        Bound::Exact => return hit.score,
                        Bound::Alpha if hit.score <= alpha => return alpha,
                        Bound::Beta if hit.score >= beta => return beta,
                        _ => (),
                    }
                    tt_move = hit.m;
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

        if !is_check && !pv_node && ply > 0 {
            let static_eval = evaluate(refs.pos);
            if null_move && depth >= 3 && static_eval >= beta {
                if let Ok(new_pos) = refs.pos.clone().swap_turn() {
                    *refs.pos = new_pos;
                }
                let score = -Search::alpha_beta(refs, depth - 3, -beta, -beta + 1, false);
                if let Ok(new_pos) = refs.pos.clone().swap_turn() {
                    *refs.pos = new_pos;
                }

                if score >= beta {
                    return beta;
                }
            }
        }

        let mut legal_moves = refs.pos.legal_moves();
        Search::sort_moves(
            &mut legal_moves,
            &refs.search_info.pv[ply][ply],
            &tt_move,
            refs,
        );
        if legal_moves.len() == 1 {
            depth += 1;
        }

        let mut best_move: Option<Move> = None;
        let mut bound = Bound::Alpha;
        let mut fail_low = true;
        for (moves_searched, m) in (&legal_moves).into_iter().enumerate() {
            let prev_pos = refs.pos.clone();
            refs.pos.play_unchecked(m);
            refs.incr_rep();

            refs.search_info.ply += 1;
            refs.search_info.prev_move[ply + 1] = Some(m.clone());

            let mut score;

            if moves_searched == 0 {
                // We always search full depth on hypothesis best move
                score = -Search::alpha_beta(refs, depth - 1, -beta, -alpha, true);
            } else {
                // Late Move Reductions - trying to fail low
                if moves_searched >= 4
                    && ply >= 3
                    && !m.is_capture()
                    && !m.is_promotion()
                    && !is_check
                    && Some(m) != refs.search_info.killer_moves1[ply].as_ref()
                    && Some(m) != refs.search_info.killer_moves2[ply].as_ref()
                {
                    score = -Search::alpha_beta(refs, depth - 2, -alpha - 1, -alpha, true);
                } else {
                    // When we don't do LMR we don't fail low
                    score = alpha + 1;
                }
                if score > alpha {
                    score = -Search::alpha_beta(refs, depth - 1, -alpha - 1, -alpha, true);
                    if score > alpha && score < beta {
                        // We found a better move so re-search
                        score = -Search::alpha_beta(refs, depth - 1, -beta, -alpha, true)
                    }
                }
            }

            refs.decr_rep();
            *refs.pos = prev_pos;
            refs.search_info.ply -= 1;

            if score >= beta {
                refs.tt
                    .write(hash.0, depth, score, Bound::Beta, Some(m.clone()), ply);

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
                best_move = Some(m.clone());
                bound = Bound::Exact;

                refs.search_info.pv[ply].fill(None);
                refs.search_info.pv[ply][ply] = best_move.clone();
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
        refs.tt.write(hash.0, depth, alpha, bound, best_move, ply);

        alpha
    }
}
