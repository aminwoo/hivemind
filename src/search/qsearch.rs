use super::{defs::SearchRefs, Search};
use crate::search::eval;
use shakmaty::Position;

impl Search {
    pub fn qsearch(refs: &mut SearchRefs, mut alpha: i16, beta: i16) -> i16 {
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
        let mut score = eval::evaluate(refs.pos);
        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }

        let mut captures = refs.pos.capture_moves();
        Search::sort_moves(&mut captures, &None, refs);

        for m in &captures {
            let prev_pos = refs.pos.clone();
            refs.pos.play_unchecked(m);
            score = -Search::qsearch(refs, -beta, -alpha);
            *refs.pos = prev_pos;

            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }
        alpha
    }
}
