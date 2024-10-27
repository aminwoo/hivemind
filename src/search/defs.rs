use crate::engine::transposition::{SearchData, TT};
use shakmaty::{Chess, Move, MoveList, Position, Role, Square};
use shakmaty_syzygy::{Tablebase, Wdl};
use std::time::Instant;

const MAX_HISTORY: i16 = 3000;

pub struct SearchResult {
    pub depth: i8,
    pub time: u128,
    pub cp: i16,
    pub nodes: usize,
    pub nps: usize,
    pub pv: MoveList,
}

pub struct SearchInfo {
    start_time: Option<Instant>,
    pub nodes: usize,
    pub cp: i16,
    pub ply: i8,
    pub tt_hits: usize,
    pub best_move: Option<Move>,
    pub killer_moves1: Vec<Option<Move>>,
    pub killer_moves2: Vec<Option<Move>>,
    pub history: [[i16; 64]; 7],
    pub terminated: bool,
}

impl SearchInfo {
    pub fn new() -> Self {
        Self {
            start_time: None,
            nodes: 0,
            cp: 0,
            ply: 0,
            tt_hits: 0,
            best_move: None,
            killer_moves1: vec![None; 128],
            killer_moves2: vec![None; 128],
            history: [[0; 64]; 7],
            terminated: false,
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

    pub fn update_history(&mut self, role: Role, to: Square, value: i16) {
        let clamped_value = value.clamp(-MAX_HISTORY, MAX_HISTORY);
        self.history[role as usize][to as usize] += clamped_value
            - self.history[role as usize][to as usize] * clamped_value.abs() / MAX_HISTORY;
    }
}

pub struct SearchParams {
    pub depth: i8,
    pub search_time: u128,
}

pub struct SearchRefs<'a> {
    pub pos: &'a mut Chess,
    pub search_params: &'a mut SearchParams,
    pub search_info: &'a mut SearchInfo,
    pub tt: &'a mut TT<SearchData>,
    pub tt_enabled: bool,
    pub tb: &'a mut Tablebase<Chess>,
}
