mod about;

pub mod transposition;

use crate::search::Search;
use shakmaty::{
    zobrist::{Zobrist64, ZobristHash},
    EnPassantMode, Position,
};
use std::collections::HashMap;
use std::io;
use std::sync::{Arc, Mutex};

use shakmaty::uci::UciMove;
use shakmaty::{fen::Fen, CastlingMode, Chess};

use transposition::{SearchData, TT};

pub struct Engine {
    pos: Arc<Mutex<Chess>>,
    repetitions: Arc<Mutex<Vec<Zobrist64>>>,
    search: Search,
    tt_search: Arc<Mutex<TT<SearchData>>>,
}

impl Engine {
    pub fn new() -> Self {
        let tt_search: Arc<Mutex<TT<SearchData>>> =
            Arc::new(Mutex::new(TT::<SearchData>::new(1024)));
        Engine {
            pos: Arc::new(Mutex::new(Chess::new())),
            repetitions: Arc::new(Mutex::new(Vec::new())),
            search: Search::new(),
            tt_search,
        }
    }

    pub fn run(&mut self) {
        self.search.init(
            Arc::clone(&self.pos),
            Arc::clone(&self.tt_search),
            Arc::clone(&self.repetitions),
        );
        self.print_logo();
        self.print_about();

        let mut cmd = String::new();
        let mut quit = false;
        while !quit {
            io::stdin()
                .read_line(&mut cmd)
                .expect("Failed to read command");

            cmd = cmd.trim_end().to_string();

            if cmd == "quit" {
                quit = true;
                self.search.send("quit".to_string());
            }
            if cmd == "uci" {
                println!("uciok");
            }
            if cmd == "isready" {
                println!("readyok");
            }
            if cmd == "ucinewgame" {
                self.tt_search.lock().unwrap().clear();
            }

            if cmd.starts_with("position") {
                enum Options {
                    Nothing,
                    Fen,
                    Moves,
                }
                let mut fen_string = String::from("");
                let mut moves: Vec<String> = Vec::new();
                let mut skip_fen = false;
                let mut option = Options::Nothing;
                let tokens: Vec<&str> = cmd.split_whitespace().collect();
                for token in tokens {
                    match token {
                        "position" => (),
                        "startpos" => skip_fen = true,
                        "fen" => option = Options::Fen,
                        "moves" => option = Options::Moves,
                        _ => match option {
                            Options::Nothing => (),
                            Options::Fen => {
                                fen_string.push_str(token);
                                fen_string.push(' ');
                            }
                            Options::Moves => moves.push(token.to_string()),
                        },
                    }
                }

                if skip_fen {
                    fen_string =
                        String::from("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                }
                let fen: Fen = fen_string.parse().unwrap();
                let mut pos_guard = self.pos.lock().unwrap();
                *pos_guard = fen.into_position(CastlingMode::Standard).unwrap();

                let mut repetitions_guard = self.repetitions.lock().unwrap();
                let hash = pos_guard.zobrist_hash(EnPassantMode::Legal);
                repetitions_guard.push(hash);

                for mv in moves {
                    let uci: UciMove = mv.parse().unwrap();
                    let m = uci.to_move(&*pos_guard).unwrap();
                    pos_guard.play_unchecked(&m);
                    let hash = pos_guard.zobrist_hash(EnPassantMode::Legal);
                    repetitions_guard.push(hash);
                }
            }

            if cmd.starts_with("go") {
                self.search.send("go".to_string());
            }

            cmd = String::new();
        }
    }
}
