use super::{defs::SearchRefs, Search};
use crate::types::Score;
use shakmaty::{Move, Position};

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
                score = Search::alpha_beta(refs, depth, -Score::INFINITY, Score::INFINITY, true);
            } else {
                score = Search::alpha_beta(refs, depth, alpha, beta, true);
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
                nodes / elapsed as usize * 1000
            } else {
                0
            };

            print!(
                "info depth {} score cp {} nodes {} nps {} hashfull {} time {} pv ",
                depth,
                score,
                nodes,
                nps,
                refs.tt.hashfull(),
                elapsed
            );
            for m in refs.search_info.pv[0].iter().flatten() {
                let uci = m.to_uci(refs.pos.castles().mode());
                print!("{} ", uci);
            }
            println!();

            best_move = refs.search_info.pv[0][0].clone();
            depth += 1;
        }
        best_move
    }
}
