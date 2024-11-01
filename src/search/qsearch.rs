use super::sorting::see;
use super::{defs::SearchRefs, Search};
use crate::search::eval;
use shakmaty::Position;

impl Search {
    pub fn qsearch(refs: &mut SearchRefs, mut alpha: i32, beta: i32) -> i32 {
        if (refs.search_info.nodes & 2047) == 0
            && refs.search_info.elapsed() > refs.search_params.search_time
        {
            refs.search_info.terminated = true;
            return 0;
        }
        if refs.search_info.elapsed() > refs.search_params.search_time {
            refs.search_info.terminated = true;
            return 0;
        }
        refs.search_info.nodes += 1;

        let is_check = refs.pos.is_check();
        let mut score;
        let eval = eval::evaluate(refs.pos);

        if eval >= beta {
            return beta;
        }
        if eval > alpha {
            alpha = eval;
        }

        let mut captures = refs.pos.capture_moves();
        Search::sort_moves(&mut captures, &None, &None, refs);

        for m in &captures {
            if !is_check && see(refs, m) < 0 {
                continue;
            }

            let prev_pos = refs.pos.clone();
            refs.pos.play_unchecked(m);
            score = -Search::qsearch(refs, -beta, -alpha);
            *refs.pos = prev_pos;

            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }

        alpha
    }
}
