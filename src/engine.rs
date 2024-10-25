mod about;

use crate::search::Search;
use shakmaty::Chess;
use std::io;

pub struct Engine {
    search: Search,
}

impl Engine {
    pub fn new() -> Self {
        Engine {
            search: Search::new(),
        }
    }

    pub fn run(&mut self) {
        //Self::go_handler();
        //
        self.print_logo();
        self.print_about();

        self.search.init();

        let mut cmd = String::new();
        let mut quit = false;
        while !quit {
            io::stdin()
                .read_line(&mut cmd)
                .expect("Failed to read command");

            cmd = cmd.trim_end().to_string();

            if cmd == "quit" {
                quit = true;
            }
            cmd = String::new();
        }
    }
}
