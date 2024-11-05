use super::history::History;
use crate::board::Board;
use crate::transposition::TranspositionTable;
use crate::types::parameters::Parameters;
use crate::types::MAX_PLY;
use shakmaty::Move;
use std::time::Instant;

pub struct SearchInfo {
    start_time: Option<Instant>,
    pub nodes: usize,
    pub sel_depth: usize,
    pub cp: i32,
    pub killers: Vec<Option<Move>>,
    pub terminated: bool,
    pub pv: Vec<Vec<Option<Move>>>,
    pub pv_length: [usize; MAX_PLY],
    pub history: History,
}

impl SearchInfo {
    pub fn new() -> Self {
        Self {
            start_time: None,
            nodes: 0,
            sel_depth: 0,
            cp: 0,
            killers: vec![None; MAX_PLY],
            terminated: false,
            pv: vec![vec![None; MAX_PLY]; MAX_PLY],
            pv_length: [0; MAX_PLY],
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
    pub board: &'a mut Board,
    pub params: Parameters,
    pub search_params: &'a mut SearchParams,
    pub search_info: &'a mut SearchInfo,
    pub tt: &'a mut TranspositionTable,
    pub tt_enabled: bool,
}
