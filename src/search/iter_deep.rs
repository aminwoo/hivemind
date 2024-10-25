use super::Search;
use shakmaty::{Move, MoveList, Position};

impl Search {
    pub fn iterative_deepening<T: Position + Clone>(pos: &mut T, max_depth: i8) -> (Move, u32) {
        let mut depth = 1;
        let mut best_move: Option<Move> = None;

        let mut node_count = 0;
        let mut root_pv = MoveList::new();

        let alpha = -10000.0;
        let beta = 10000.0;

        while depth <= max_depth {
            Search::alpha_beta(pos, depth, alpha, beta, &mut root_pv, &mut node_count);

            print!(
                "info depth {} score cp {} nodes {} nps {} time {} pv ",
                depth, 0, node_count, 0, 0
            );

            for mv in &root_pv {
                let uci = mv.to_uci(pos.castles().mode());
                print!("{} ", uci);
            }
            println!();

            best_move = Some(root_pv[0].clone());
            depth += 1;
        }

        (best_move.unwrap(), node_count)
    }
}
