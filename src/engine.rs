mod about;

use crate::search::Search;
use crate::transposition::TranspositionTable;
use shakmaty::{
    zobrist::{Zobrist64, ZobristHash},
    EnPassantMode, Position,
};
use std::io;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use shakmaty::uci::UciMove;
use shakmaty::{fen::Fen, perft, CastlingMode, Chess};

pub struct Engine {
    pos: Arc<Mutex<Chess>>,
    repetitions: Arc<Mutex<Vec<Zobrist64>>>,
    search: Search,
    tt_search: Arc<Mutex<TranspositionTable>>,
}

impl Engine {
    pub fn new() -> Self {
        let tt_search: Arc<Mutex<TranspositionTable>> =
            Arc::new(Mutex::new(TranspositionTable::default()));
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
                repetitions_guard.clear();
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
                if let Some(token) = cmd.split_whitespace().nth(1) {
                    if token == "movetime" {
                        self.search.send(cmd[3..].to_string());
                    }
                }
                self.search.send("go".to_string());
            }
            if cmd == "stop" {
                self.search.send("stop".to_string());
            }
            if cmd.starts_with("perft") {
                if let Some(depth) = cmd.split_whitespace().nth(1) {
                    if let Ok(depth) = depth.parse::<u32>() {
                        let pos_guard = self.pos.lock().unwrap();
                        let now = Instant::now();
                        let nodes = perft(&pos_guard.clone(), depth);

                        let elapsed_time = now.elapsed();

                        println!(
                            "Nodes {} | Elapsed {:.3}s | NPS {:.3} kN/s",
                            nodes,
                            elapsed_time.as_secs_f64(),
                            nodes as f64 / (elapsed_time.as_secs_f64() * 1000.0)
                        )
                    }
                }
            }

            cmd = String::new();
        }
    }
}
