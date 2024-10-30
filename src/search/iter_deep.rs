use super::{defs::SearchRefs, Search};
use crate::types::Score;
use shakmaty::{Move, Position};

impl Search {
    pub fn iterative_deepening(refs: &mut SearchRefs) -> Option<Move> {
        let mut depth = 1;
        let mut alpha = -Score::INFINITY;
        let mut beta = Score::INFINITY;
        let aspiration_window = 25;
        let mut best_move: Option<Move> = None;

        refs.search_info.start();
        while depth <= refs.search_params.depth {
            let cp = Search::alpha_beta(refs, depth, alpha, beta, true);
            if refs.search_info.terminated {
                break;
            }

            if cp <= alpha || cp >= beta {
                alpha = -Score::INFINITY;
                beta = Score::INFINITY;
                continue;
            } else {
                alpha = cp - aspiration_window;
                beta = cp + aspiration_window;
            }
            refs.search_info.cp = cp;

            let nodes = refs.search_info.nodes;
            let elapsed = refs.search_info.elapsed();
            let nps = if elapsed > 0 {
                nodes / elapsed as usize * 1000
            } else {
                0
            };

            print!(
                "info depth {} score cp {} nodes {} nps {} tbhits {} time {} pv ",
                depth, cp, nodes, nps, refs.search_info.tt_hits, elapsed
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
