use super::{defs::SearchRefs, Search};
use crate::search::eval;
use shakmaty::Position;

impl Search {
    pub fn qsearch(refs: &mut SearchRefs, mut alpha: i16, beta: i16) -> i16 {
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
        let scores = Search::score_moves(&mut captures, &None, refs);

        for i in 0..captures.len() {
            Search::pick_move(&mut captures, i, scores);
            let mv = captures.get(i).unwrap();

            let prev_pos = refs.pos.clone();
            refs.pos.play_unchecked(mv);
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
