use super::{defs::SearchRefs, Search};
use crate::types::Score;
use shakmaty::Move;

impl Search {
    pub fn iterative_deepening(refs: &mut SearchRefs) -> Option<Move> {
        let mut depth = 1;
        let mut alpha = -Score::INFINITY;
        let mut beta = Score::INFINITY;
        let delta = 25;
        let mut best_move: Option<Move> = None;
        let mut score;

        refs.search_info.start();
        while depth <= refs.search_params.depth {
            if depth <= 5 {
                score = Search::alpha_beta(refs, depth, -Score::INFINITY, Score::INFINITY);
            } else {
                score = Search::alpha_beta(refs, depth, alpha, beta);
            }
            if refs.search_info.terminated {
                break;
            }

            if score <= alpha || score >= beta {
                alpha = -Score::INFINITY;
                beta = Score::INFINITY;
                continue;
            }
            alpha = (score - delta).max(-Score::INFINITY);
            beta = (score + delta).min(Score::INFINITY);

            refs.search_info.cp = score;

            let nodes = refs.search_info.nodes;
            let elapsed = refs.search_info.elapsed();
            let nps = if elapsed > 0 {
                (nodes as f64 / elapsed as f64) * 1000.0
            } else {
                0.0
            };
            let sel_depth = refs.search_info.sel_depth;

            print!(
                "info depth {} seldepth {} score cp {} nodes {} nps {:.0} hashfull {} time {} pv ",
                depth,
                sel_depth,
                score,
                nodes,
                nps,
                refs.tt.hashfull(),
                elapsed
            );
            for mv in refs.search_info.pv[0].iter().flatten() {
                let uci = refs.board.to_uci(mv);
                print!("{} ", uci);
            }
            println!();

            best_move = refs.search_info.pv[0][0].clone();
            depth += 1;
        }
        best_move
    }
}
