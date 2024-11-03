use super::{defs::SearchRefs, sorting::see, Search};
use crate::types::MAX_PLY;

impl Search {
    pub fn qsearch(refs: &mut SearchRefs, mut alpha: i32, beta: i32) -> i32 {
        if (refs.search_info.nodes & 2047) == 0
            && refs.search_info.elapsed() > refs.search_params.search_time
        {
            refs.search_info.terminated = true;
            return 0;
        }
        let ply = refs.search_info.ply as usize;
        refs.search_info.nodes += 1;
        refs.search_info.sel_depth = refs.search_info.sel_depth.max(ply);

        let eval = refs.board.evaluate();
        if eval >= beta || ply >= MAX_PLY - 1 {
            return beta;
        }
        if eval > alpha {
            alpha = eval;
        }

        let mut best_score = eval;

        let mut captures = refs.board.capture_moves();
        Search::sort_moves(&mut captures, &None, &None, refs);

        for mv in &captures {
            refs.board.make_move::<false>(mv);
            let score = -Search::qsearch(refs, -beta, -alpha);
            refs.board.undo_move();

            if score > best_score {
                best_score = score;
                if score > alpha {
                    alpha = score;
                }
            }

            if alpha >= beta {
                break;
            }
        }

        best_score
    }
}
