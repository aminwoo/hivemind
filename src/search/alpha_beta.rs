use super::Search;
use shakmaty::{Move, MoveList, Position};

impl Search {
    pub fn alpha_beta<T: Position + Clone>(
        pos: &mut T,
        depth: i8,
        mut alpha: f32,
        beta: f32,
        pv: &mut MoveList,
        node_count: &mut u32,
    ) -> f32 {
        *node_count += 1;

        if depth <= 0 {
            return Search::qsearch(pos, 99, alpha, beta, node_count);
        }

        let mut legal_moves = pos.legal_moves();
        Search::sort_moves(&mut legal_moves);

        for mv in &legal_moves {
            let prev_pos = pos.clone();
            pos.play_unchecked(mv);

            let mut node_pv: MoveList = MoveList::new();
            let score =
                -Search::alpha_beta(pos, depth - 1, -beta, -alpha, &mut node_pv, node_count);

            *pos = prev_pos;

            if score >= beta {
                return beta;
            }

            if score > alpha {
                alpha = score;
                pv.clear();
                pv.push(mv.clone());
                pv.extend(node_pv.iter().cloned());
            }
        }
        alpha
    }
}
