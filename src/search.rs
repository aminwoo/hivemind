mod alpha_beta;
mod eval;
mod iter_deep;
mod qsearch;
mod sorting;
use std::time::Instant;

use shakmaty::variant::{self, Crazyhouse};
use shakmaty::{Chess, Position};

pub struct Search {}

impl Search {
    pub fn new() -> Self {
        Self {}
    }

    pub fn init(&mut self) {
        let quit = false;

        let mut pos = Chess::default();

        let depth = 7;

        let start = Instant::now();
        let (best_move, node_count) = Search::iterative_deepening(&mut pos, depth);
        let duration = start.elapsed();
        let nps = node_count as f64 / duration.as_secs_f64();
        println!("bestmove {}", best_move.to_uci(pos.castles().mode()));
        println!("Nodes searched: {}", node_count);
        println!("NPS: {:.2?}", nps);
        println!("Time taken: {:.2?}", duration);
    }
}
