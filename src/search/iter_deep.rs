use super::{defs::SearchRefs, Search};
use crate::types::parameters::*;
use crate::types::Score;
use shakmaty::Move;

impl Search {
    pub fn aspiration_search(refs: &mut SearchRefs, mut score: i32, depth: i32) -> i32 {
        refs.board.set_ply(0);
        if depth <= aspiration_depth() {
            return Search::alpha_beta(refs, depth, -Score::INFINITY, Score::INFINITY);
        }

        let mut delta = (aspiration_delta() - depth).max(10);
        let mut alpha = (score - delta).max(-Score::INFINITY);
        let mut beta = (score + delta).min(Score::INFINITY);
        let mut fail_high_count = 0;

        loop {
            let adjusted_depth = (depth - fail_high_count).max(1);
            score = Search::alpha_beta(refs, adjusted_depth, alpha, beta);

            if refs.search_info.terminated {
                return 0;
            }

            if score <= alpha {
                alpha = (alpha - delta).max(-Score::INFINITY);
                beta = (alpha + beta) / 2;
                fail_high_count = 0;
            } else if score >= beta {
                beta = (beta + delta).min(Score::INFINITY);
                fail_high_count += 1;
            } else {
                return score;
            }

            delta += delta / 2;
        }
    }

    pub fn iterative_deepening(refs: &mut SearchRefs) -> Option<Move> {
        let mut best_move: Option<Move> = None;
        let mut score = 0;

        refs.search_info.start();
        for depth in 1..refs.search_params.depth {
            score = Search::aspiration_search(refs, score, depth);
            if refs.search_info.terminated {
                break;
            }

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
        }
        best_move
    }
}
