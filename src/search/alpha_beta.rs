use crate::search::sorting::see;
use crate::transposition::{Bound, Entry};
use crate::types::{parameters::*, Score, MAX_PLY};

use super::{defs::SearchRefs, Search};
use shakmaty::{Move, MoveList};

impl Search {
    pub fn alpha_beta(refs: &mut SearchRefs, mut depth: i32, mut alpha: i32, mut beta: i32) -> i32 {
        let ply = refs.board.ply();
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
        let mut best_move: Option<&Move> = None;

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

        let mut tt_move: Option<Move> = None;
        let hit = refs.tt.read(refs.board.get_hash(), ply);
        if let Some(hit) = &hit {
            if !pv_node && hit.valid_cutoff(alpha, beta, depth) {
                return hit.score;
            }
            tt_move = hit.mv.clone();
        }

        // Internal Iterative Reductions
        if !is_root && tt_move.is_none() && depth >= iir_depth() {
            depth -= 1;
        }

        if in_check {
            depth += 1;
        }

        refs.search_info.nodes += 1;
        refs.search_info.sel_depth = refs.search_info.sel_depth.max(ply);
        refs.search_info.pv_length[ply] = ply;

        let eval = refs.board.evaluate();
        let improving = refs.board.is_improving();
        refs.board
            .set_eval(ply, if in_check { -Score::INFINITY } else { eval });

        if !in_check && !pv_node && !is_root {
            // Reverse Futility Pruning
            if depth < rfp_depth() && eval - rfp_margin() * (depth - i32::from(improving)) > beta {
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
            // Null move pruning
            if !refs.board.is_last_move_null()
                && depth >= 4
                && eval > beta
                && refs.board.has_non_pawn_material()
            {
                let r = 3 + depth / 3 + ((eval - beta) / 200).min(4);

                refs.board.make_null_move();
                let score = -Search::alpha_beta(refs, depth - r, -beta, -beta + 1);
                refs.board.undo_null_move();

                if score >= beta {
                    return beta;
                }
            }
        }

        let mut moves = refs.board.legal_moves();
        Search::sort_moves(&mut moves, &refs.search_info.pv[ply][ply], &tt_move, refs);

        for (moves_searched, mv) in (&moves).into_iter().enumerate() {
            if !is_root && moves_searched > 0 && alpha > -Score::MATE_BOUND {
                // Futility Pruning
                if !pv_node
                    && !in_check
                    && !mv.is_capture()
                    && depth <= fp_depth()
                    && eval + fp_margin() * depth + fp_fixed_margin() < alpha
                {
                    break;
                }
                // Late Move Pruning. Leave the node after trying enough quiet moves with no success.
                if !mv.is_capture()
                    && depth <= LMP_DEPTH
                    && quiets.len() as i32 > LMP_MARGIN + depth * depth / (2 - improving as i32)
                {
                    break;
                } // Late Move Pruning. Leave the node after trying enough quiet moves with no success.
                if !mv.is_capture()
                    && depth <= LMP_DEPTH
                    && quiets.len() as i32 > LMP_MARGIN + depth * depth
                {
                    break;
                }

                // Static Exchange Evaluation Pruning. Skip moves that are losing material.
                if depth < see_depth()
                    && !see(
                        &refs.board.state(),
                        mv,
                        -[see_quiet_margin(), see_noisy_margin()][mv.is_capture() as usize] * depth,
                    )
                    .expect("Error evaluationg SEE")
                {
                    continue;
                }
            }
            refs.board.make_move::<false>(mv);
            refs.tt.prefetch(refs.board.get_hash());

            let mut score;
            if moves_searched == 0 {
                score = -Search::alpha_beta(refs, depth - 1, -beta, -alpha);
            } else {
                let reduction = Search::calculate_reduction(
                    refs,
                    pv_node,
                    mv,
                    depth,
                    moves_searched as i32,
                    improving,
                    &hit,
                );

                let mut new_depth = depth - 1;
                score = -Search::alpha_beta(refs, new_depth - reduction, -alpha - 1, -alpha);
                if alpha < score && reduction > 0 {
                    new_depth += i32::from(score > best_score + search_deeper_margin());
                    score = -Search::alpha_beta(refs, new_depth, -alpha - 1, -alpha);
                }
                if alpha < score && score < beta {
                    score = -Search::alpha_beta(refs, new_depth, -beta, -alpha);
                }
            }

            refs.board.undo_move();

            if score > best_score {
                best_score = score;
                best_move = Some(mv);

                if score > alpha {
                    alpha = score;
                    Search::update_pv(refs, best_move, ply);
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
                best_move.cloned().expect("Move should not be None"),
                captures,
                quiets,
            );
        }

        refs.tt.write(
            refs.board.get_hash(),
            depth,
            best_score,
            bound,
            best_move.cloned(),
            ply,
        );
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
            refs.search_info.history.update_capture(
                refs.board.state(),
                &best_move,
                &captures,
                depth,
            );
        } else {
            refs.search_info.killers[refs.board.ply()] = Some(best_move.clone());
            refs.search_info
                .history
                .update_main(refs.board.turn(), &best_move, &quiets, depth);
            refs.search_info
                .history
                .update_continuation(refs.board, &best_move, &quiets, depth);
        }
    }

    pub fn update_pv(refs: &mut SearchRefs, best_move: Option<&Move>, ply: usize) {
        refs.search_info.pv[ply][ply] = best_move.cloned();
        for next_ply in ply + 1..refs.search_info.pv_length[ply + 1] {
            refs.search_info.pv[ply][next_ply] = refs.search_info.pv[ply + 1][next_ply].clone();
        }
        refs.search_info.pv_length[ply] = refs.search_info.pv_length[ply + 1];
    }

    pub fn calculate_reduction(
        refs: &mut SearchRefs,
        pv_node: bool,
        mv: &Move,
        depth: i32,
        moves: i32,
        improving: bool,
        entry: &Option<Entry>,
    ) -> i32 {
        fn to_f64(v: bool) -> f64 {
            i32::from(v) as f64
        }

        if mv.is_capture() || moves < LMR_MOVES_PLAYED || depth < LMR_DEPTH {
            return 0;
        }

        // Fractional reductions
        let mut reduction = refs.params.lmr(depth, moves);

        reduction -= refs
            .search_info
            .history
            .get_main(!refs.board.turn(), mv)
            .unwrap() as f64
            / lmr_history() as f64;

        reduction -= 0.88 * to_f64(!pv_node);
        reduction -= 0.78 * to_f64(refs.board.in_check());

        reduction += 0.91
            * to_f64(
                entry
                    .as_ref()
                    .is_some_and(|e| e.mv.is_some() && e.mv.clone().unwrap().is_capture()),
            );
        reduction += 0.48 * to_f64(improving);

        // Avoid negative reductions
        (reduction as i32).clamp(0, depth)
    }
}
