use crate::transposition::{Bound, Entry};
use crate::types::parameters::*;
use crate::types::Score;

use super::{defs::SearchRefs, eval::evaluate, Search};
use shakmaty::{Move, MoveList, Position};

impl Search {
    pub fn alpha_beta(
        refs: &mut SearchRefs,
        mut depth: i32,
        mut alpha: i32,
        mut beta: i32,
        null_move: bool,
    ) -> i32 {
        if (refs.search_info.nodes & 2047) == 0
            && refs.search_info.elapsed() > refs.search_params.search_time
        {
            refs.search_info.terminated = true;
            return 0;
        }

        let ply = refs.search_info.ply as usize;
        let is_root = ply == 0;
        let pv_node = beta - alpha > 1;

        let mut best_move: Option<Move> = None;
        let mut bound = Bound::Alpha;
        let mut fail_low = true;

        let mut captures = MoveList::default();
        let mut quiets = MoveList::default();

        if !is_root {
            // Draw Detection
            if refs.three_fold() {
                return Score::DRAW;
            }

            // Mate Distance Pruning
            alpha = alpha.max(-Score::MATE + ply as i32);
            beta = beta.min(Score::MATE - (ply as i32) - 1);
            if alpha >= beta {
                return alpha;
            }
        }

        if depth <= 0 {
            return Search::qsearch(refs, alpha, beta);
        }

        let hash = refs.get_hash();
        let mut tt_move: Option<Move> = None;
        let hit = refs.tt.read(hash.0, ply);
        if let Some(hit) = hit {
            /*    if !pv_node && hit.depth >= depth {
                match hit.bound {
                    Bound::Exact => return hit.score,
                    Bound::Alpha if hit.score <= alpha => return alpha,
                    Bound::Beta if hit.score >= beta => return beta,
                    _ => (),
                }
            }*/
            tt_move = hit.m;
        }

        // Internal Iterative Reductions
        if !is_root && tt_move.is_none() && depth >= iir_depth() {
            depth -= 1;
        }

        refs.search_info.pv_length[ply] = ply;
        let is_check = refs.pos.is_check();
        if is_check {
            depth += 1;
        }
        refs.search_info.nodes += 1;

        let eval = evaluate(refs.pos);
        if !is_check && !pv_node && !is_root {
            // Reverse Futility Pruning
            if depth < rfp_depth() && eval - rfp_margin() * depth >= beta {
                return eval;
            }
            // Razoring
            if depth <= razoring_depth()
                && eval + razoring_margin() * depth + razoring_fixed_margin() <= alpha
            {
                let score = Search::qsearch(refs, alpha, beta);
                if score <= alpha {
                    return score;
                }
            }
            // Null Move Pruning
            if null_move && depth >= 3 && eval >= beta {
                if let Ok(new_pos) = refs.pos.clone().swap_turn() {
                    let prev_pos = refs.pos.clone();
                    *refs.pos = new_pos;
                    let score = -Search::alpha_beta(refs, depth - 3, -beta, -beta + 1, false);
                    *refs.pos = prev_pos;
                    if score >= beta {
                        return beta;
                    }
                }
            }
        }

        let mut legal_moves = refs.pos.legal_moves();
        if legal_moves.len() == 1 {
            depth += 1;
        }

        Search::sort_moves(
            &mut legal_moves,
            &refs.search_info.pv[ply][ply],
            &tt_move,
            refs,
        );

        for (moves_searched, m) in (&legal_moves).into_iter().enumerate() {
            if !is_root && moves_searched > 0 && alpha > -Score::MATE_BOUND {
                // Futility Pruning
                if !pv_node
                    && !is_check
                    && !m.is_capture()
                    && depth <= fp_depth()
                    && eval + fp_margin() * depth + fp_fixed_margin() < alpha
                {
                    break;
                }
            }

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
                // Late Move Reductions - try to fail low
                if moves_searched >= LMR_MOVES_PLAYED
                    && depth >= LMR_DEPTH
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
                Search::update_ordering_heuristics(refs, depth, m.clone(), captures, quiets);
                return beta;
            } else if m.is_capture() {
                refs.search_info
                    .update_capture_history(m.role(), m.to(), m.capture().unwrap(), -1);
            }

            if score > alpha {
                alpha = score;
                best_move = Some(m.clone());
                fail_low = false;
                bound = Bound::Exact;

                // Update PV Table
                Search::update_pv(refs, best_move.clone(), ply);
            } else if is_root && fail_low {
                return -Score::INFINITY;
            }

            if m.is_capture() {
                captures.push(m.clone());
            } else {
                quiets.push(m.clone());
            }
        }

        if legal_moves.is_empty() {
            if is_check {
                return -Score::INFINITY + (refs.search_info.ply as i32);
            } else {
                return Score::DRAW;
            }
        }
        refs.tt.write(hash.0, depth, alpha, bound, best_move, ply);
        alpha
    }

    pub fn update_ordering_heuristics(
        refs: &mut SearchRefs,
        depth: i32,
        best_move: Move,
        captures: MoveList,
        quiets: MoveList,
    ) {
        if best_move.is_capture() {
            refs.search_info
                .history
                .update_capture(refs.pos.clone(), best_move, captures, depth);
        } else {
            refs.search_info.killers[refs.search_info.ply as usize] = Some(best_move.clone());
            refs.search_info
                .history
                .update_main(refs.pos.turn(), best_move, quiets, depth);
        }
    }

    pub fn update_pv(refs: &mut SearchRefs, best_move: Option<Move>, ply: usize) {
        refs.search_info.pv[ply].fill(None);
        refs.search_info.pv[ply][ply] = best_move.clone();
        for next_ply in ply + 1..refs.search_info.pv_length[ply + 1] {
            refs.search_info.pv[ply][next_ply] = refs.search_info.pv[ply + 1][next_ply].clone();
        }
        refs.search_info.pv_length[ply] = refs.search_info.pv_length[ply + 1];
    }
}
