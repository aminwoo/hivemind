use crate::transposition::Bound;
use crate::types::{parameters::*, Score, MAX_PLY};

use super::{defs::SearchRefs, Search};
use shakmaty::{Move, MoveList};

impl Search {
    pub fn alpha_beta(refs: &mut SearchRefs, mut depth: i32, mut alpha: i32, mut beta: i32) -> i32 {
        let ply = refs.search_info.ply as usize;
        refs.search_info.pv[ply].fill(None);

        if (refs.search_info.nodes & 2047) == 0
            && refs.search_info.elapsed() > refs.search_params.search_time
        {
            refs.search_info.terminated = true;
            return 0;
        }

        let is_root = ply == 0;
        let pv_node = beta - alpha > 1;
        let original_alpha = alpha;
        let in_check = refs.board.in_check();

        let mut best_score: i32 = -Score::INFINITY;
        let mut best_move: Option<Move> = None;
        let mut fail_low = true;

        let mut captures = MoveList::default();
        let mut quiets = MoveList::default();

        if !is_root {
            // Draw Detection
            if refs.board.three_fold() {
                return Score::DRAW;
            }

            // Mate Distance Pruning
            alpha = alpha.max(-Score::MATE + ply as i32);
            beta = beta.min(Score::MATE - (ply as i32) - 1);
            if alpha >= beta {
                return alpha;
            }
        }

        if ply >= MAX_PLY - 1 {
            return refs.board.evaluate();
        }

        if depth <= 0 && !in_check {
            return Search::qsearch(refs, alpha, beta);
        }
        depth = depth.max(0);

        refs.search_info.nodes += 1;
        refs.search_info.sel_depth = refs.search_info.sel_depth.max(ply);
        refs.search_info.pv_length[ply] = ply;

        let mut moves = refs.board.legal_moves();
        Search::sort_moves(&mut moves, &refs.search_info.pv[ply][ply], &None, refs);

        for (moves_searched, mv) in (&moves).into_iter().enumerate() {
            refs.board.make_move::<false>(mv);
            refs.search_info.ply += 1;

            let mut score;
            if moves_searched == 0 {
                // We always search full depth on hypothesis best move
                score = -Search::alpha_beta(refs, depth - 1, -beta, -alpha);
            } else {
                // Late Move Reductions - try to fail low
                if moves_searched >= LMR_MOVES_PLAYED
                    && depth >= LMR_DEPTH
                    && ply >= 3
                    && !mv.is_capture()
                    && !mv.is_promotion()
                    && !in_check
                    && Some(mv) != refs.search_info.killers[ply].as_ref()
                {
                    score = -Search::alpha_beta(refs, depth - 2, -alpha - 1, -alpha);
                } else {
                    // When we don't do LMR we don't fail low
                    score = alpha + 1;
                }
                if score > alpha {
                    score = -Search::alpha_beta(refs, depth - 1, -alpha - 1, -alpha);
                    if score > alpha && score < beta {
                        // We found a better move so re-search
                        score = -Search::alpha_beta(refs, depth - 1, -beta, -alpha);
                    }
                }
            }

            refs.board.undo_move();
            refs.search_info.ply -= 1;

            if score > best_score {
                best_score = score;
                best_move = Some(mv.clone());

                if score > alpha {
                    alpha = score;
                    Search::update_pv(refs, best_move.clone(), ply);
                }
            }

            if alpha >= beta {
                break;
            }

            if mv.is_capture() {
                captures.push(mv.clone());
            } else {
                quiets.push(mv.clone());
            }
        }

        if moves.is_empty() {
            return if in_check {
                Score::mated_in(ply)
            } else {
                Score::DRAW
            };
        }

        let bound = match best_score {
            s if s <= original_alpha => Bound::Alpha,
            s if s >= beta => Bound::Beta,
            _ => Bound::Exact,
        };
        if bound == Bound::Beta {
            Search::update_ordering_heuristics(
                refs,
                depth,
                best_move.clone().expect("Move should not be None"),
                captures,
                quiets,
            );
        }

        best_score
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
                .update_capture(refs.board.chess(), best_move, captures, depth);
        } else {
            refs.search_info.killers[refs.search_info.ply as usize] = Some(best_move.clone());
            refs.search_info
                .history
                .update_main(refs.board.turn(), best_move, quiets, depth);
        }
    }

    pub fn update_pv(refs: &mut SearchRefs, best_move: Option<Move>, ply: usize) {
        refs.search_info.pv[ply][ply] = best_move.clone();
        for next_ply in ply + 1..refs.search_info.pv_length[ply + 1] {
            refs.search_info.pv[ply][next_ply] = refs.search_info.pv[ply + 1][next_ply].clone();
        }
        refs.search_info.pv_length[ply] = refs.search_info.pv_length[ply + 1];
    }
}
