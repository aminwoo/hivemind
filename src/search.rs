mod alpha_beta;
mod defs;
mod eval;
mod iter_deep;
mod qsearch;
mod sorting;
use crate::engine::transposition::{SearchData, TT};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crossbeam_channel::unbounded;
use crossbeam_channel::Sender;

use defs::{SearchInfo, SearchParams, SearchRefs};
use shakmaty::{Chess, Move, Position};
use shakmaty_syzygy::Tablebase;

pub struct Search {
    handle: Option<JoinHandle<()>>,
    sender: Option<Sender<String>>,
}

impl Search {
    pub fn new() -> Self {
        Self {
            handle: None,
            sender: None,
        }
    }

    pub fn init(&mut self, mtx_pos: Arc<Mutex<Chess>>, mtx_tt: Arc<Mutex<TT<SearchData>>>) {
        let (s, r) = unbounded::<String>();
        let h = thread::spawn(move || {
            let mut quit = false;
            let mut halt = true;

            let mut best_move: Option<Move>;
            while !quit {
                let cmd = r.recv().unwrap();
                let pos = mtx_pos.lock().unwrap();
                let mut tt = mtx_tt.lock().unwrap();

                match cmd.as_str() {
                    "go" => halt = false,
                    "stop" => halt = true,
                    "quit" => quit = true,
                    _ => (),
                }

                if !halt && !quit {
                    let mut search_info = SearchInfo::new();
                    let mut search_params = SearchParams {
                        depth: 10,
                        search_time: 15000,
                    };
                    let mut tables: Tablebase<Chess> = Tablebase::new();
                    tables.add_directory("tables/chess").unwrap();

                    let mut search_refs = SearchRefs {
                        pos: &mut pos.clone(),
                        search_params: &mut search_params,
                        search_info: &mut search_info,
                        tt: &mut tt,
                        tt_enabled: true,
                        tb: &mut tables,
                    };

                    best_move = Some(Search::iterative_deepening(&mut search_refs));
                    println!(
                        "bestmove {}",
                        best_move
                            .clone()
                            .expect("Reason")
                            .to_uci(search_refs.pos.castles().mode())
                    );

                    halt = true;
                }
            }
        });

        self.handle = Some(h);
        self.sender = Some(s);
    }

    pub fn send(&mut self, cmd: String) {
        if let Some(s) = &self.sender {
            s.send(cmd).expect("Broken channel");
        }
    }
}
