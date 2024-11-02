use super::history::History;
use crate::transposition::TranspositionTable;
use shakmaty::{
    zobrist::{Zobrist64, ZobristHash},
    EnPassantMode,
};
use shakmaty::{Chess, Move, Role, Square};
use std::time::Instant;

pub struct SearchInfo {
    start_time: Option<Instant>,
    pub nodes: usize,
    pub cp: i32,
    pub ply: i8,
    pub killers: Vec<Option<Move>>,
    pub prev_move: Vec<Option<Move>>,
    pub terminated: bool,
    pub pv: Vec<Vec<Option<Move>>>,
    pub pv_length: [usize; 128],
    pub history: History,
}

impl SearchInfo {
    pub fn new() -> Self {
        Self {
            start_time: None,
            nodes: 0,
            cp: 0,
            ply: 0,
            killers: vec![None; 128],
            prev_move: vec![None; 128],
            terminated: false,
            pv: vec![vec![None; 128]; 128],
            pv_length: [0; 128],
            history: History::default(),
        }
    }

    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    pub fn elapsed(&mut self) -> u128 {
        match self.start_time {
            Some(x) => x.elapsed().as_millis(),
            None => 0,
        }
    }
}

pub struct SearchParams {
    pub depth: i32,
    pub search_time: u128,
}

pub struct SearchRefs<'a> {
    pub pos: &'a mut Chess,
    pub repetitions: &'a mut Vec<Zobrist64>,
    pub search_params: &'a mut SearchParams,
    pub search_info: &'a mut SearchInfo,
    pub tt: &'a mut TranspositionTable,
    pub tt_enabled: bool,
}

impl<'a> SearchRefs<'a> {
    pub fn three_fold(&self) -> bool {
        let mut cnt = 0;
        let hash = self.pos.zobrist_hash(EnPassantMode::Legal);
        for &h in self.repetitions.iter().rev() {
            if h == hash {
                cnt += 1;
                if cnt > 1 {
                    return true;
                }
            }
        }
        false
    }
    pub fn incr_rep(&mut self) {
        let hash = self.pos.zobrist_hash(EnPassantMode::Legal);
        self.repetitions.push(hash);
    }
    pub fn decr_rep(&mut self) {
        self.repetitions.pop();
    }

    pub fn get_hash(&self) -> Zobrist64 {
        self.pos.zobrist_hash(EnPassantMode::Legal)
    }
}
