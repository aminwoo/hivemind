mod about;

use crate::board::Board;
use crate::transposition::TranspositionTable;
use crate::{
    benchmark::{benchmark, perft},
    search::Search,
    types::Score,
};
use std::io;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use shakmaty::Chess;

pub struct Engine {
    board: Arc<Mutex<Board>>,
    search: Search,
    tt_search: Arc<Mutex<TranspositionTable>>,
}

impl Engine {
    pub fn new() -> Self {
        let tt_search: Arc<Mutex<TranspositionTable>> =
            Arc::new(Mutex::new(TranspositionTable::default()));
        Engine {
            board: Arc::new(Mutex::new(Board::starting_position())),
            search: Search::new(),
            tt_search,
        }
    }

    pub fn run(&mut self) {
        self.search
            .init(Arc::clone(&self.board), Arc::clone(&self.tt_search));
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
                let mut board_guard = self.board.lock().unwrap();
                *board_guard = Board::new(&fen_string).unwrap();

                for mv in moves {
                    board_guard.play_uci(&mv);
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
                    if let Ok(depth) = depth.parse::<i32>() {
                        let now = Instant::now();
                        let nodes = perft(&Chess::default(), depth);

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

            if cmd.starts_with("bench") {
                if let Some(depth) = cmd.split_whitespace().nth(1) {
                    if let Ok(depth) = depth.parse::<i32>() {
                        let pos = Chess::new();
                        let mut nodes = 0;
                        let now = Instant::now();

                        benchmark(&pos, -Score::INFINITY, Score::INFINITY, depth, &mut nodes);

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
            if cmd == "eval" {
                let board_guard = self.board.lock().unwrap();
                println!("{}", board_guard.evaluate());
            }

            cmd = String::new();
        }
    }
}
