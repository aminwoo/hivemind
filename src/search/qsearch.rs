use super::{defs::SearchRefs, sorting::see, Search};
use crate::transposition::Bound;
use crate::types::MAX_PLY;
use shakmaty::Move;

impl Search {
    pub fn qsearch(refs: &mut SearchRefs, mut alpha: i32, beta: i32) -> i32 {
        if (refs.search_info.nodes & 2047) == 0
            && refs.search_info.elapsed() > refs.search_params.search_time
        {
            refs.search_info.terminated = true;
            return 0;
        }
        let ply = refs.board.ply() as usize;
        refs.search_info.nodes += 1;
        refs.search_info.sel_depth = refs.search_info.sel_depth.max(ply);

        let mut tt_move: Option<Move> = None;
        let hit = refs.tt.read(refs.board.get_hash(), ply);
        if let Some(hit) = hit {
            if hit.valid_cutoff(alpha, beta, 0) {
                return hit.score;
            }
            tt_move = hit.mv;
        }

        let eval = refs.board.evaluate();
        if eval >= beta || ply >= MAX_PLY - 1 {
            return beta;
        }
        if eval > alpha {
            alpha = eval;
        }

        let in_check = refs.board.in_check();
        let mut best_move: Option<&Move> = None;
        let mut best_score = eval;

        let mut moves = refs.board.capture_moves();
        Search::sort_moves(&mut moves, &None, &tt_move, refs);

        for mv in &moves {
            if !in_check && !see(&refs.board.state(), mv, 0).expect("Error evaluating SEE") {
                continue;
            }

            refs.board.make_move::<false>(mv);
            refs.tt.prefetch(refs.board.get_hash());
            let score = -Search::qsearch(refs, -beta, -alpha);
            refs.board.undo_move();

            if score > best_score {
                best_score = score;
                best_move = Some(mv);

                alpha = alpha.max(score);
            }

            if alpha >= beta {
                break;
            }
        }

        let bound = if best_score >= beta {
            Bound::Beta
        } else {
            Bound::Alpha
        };

        refs.tt.write(
            refs.board.get_hash(),
            0,
            best_score,
            bound,
            best_move.cloned(),
            ply,
        );

        best_score
    }
}
