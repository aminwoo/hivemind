use super::Search;
use crate::search::eval;
use shakmaty::Position;

impl Search {
    pub fn qsearch<T: Position + Clone>(
        pos: &mut T,
        depth: i8,
        mut alpha: f32,
        beta: f32,
        node_count: &mut u32,
    ) -> f32 {
        *node_count += 1;
        let mut score = eval::evaluate(pos);
        if depth == 0 {
            return score;
        }
        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }

        let mut captures = pos.capture_moves();
        Search::sort_moves(&mut captures);

        for mv in &captures {
            let prev_pos = pos.clone();
            pos.play_unchecked(mv);
            score = -Search::qsearch(pos, depth - 1, -beta, -alpha, node_count);
            *pos = prev_pos;

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
