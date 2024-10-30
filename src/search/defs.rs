use crate::transposition::TranspositionTable;
use shakmaty::{
    zobrist::{Zobrist64, ZobristHash},
    EnPassantMode, Position,
};
use shakmaty::{Chess, Move, MoveList, Role, Square};
use std::time::Instant;

const MAX_HISTORY: i16 = 2000;
const MAX_CAPTURE: i16 = 1000;

pub struct SearchInfo {
    start_time: Option<Instant>,
    pub nodes: usize,
    pub cp: i16,
    pub ply: i8,
    pub tt_hits: usize,
    pub killer_moves1: Vec<Option<Move>>,
    pub killer_moves2: Vec<Option<Move>>,
    pub capture_history: [[[i16; 7]; 64]; 7],
    pub quiet_history: [[i16; 64]; 7],
    pub counter_moves: Vec<Vec<Option<Move>>>,
    pub prev_move: Vec<Option<Move>>,
    pub terminated: bool,
    pub pv: Vec<Vec<Option<Move>>>,
    pub pv_length: [usize; 128],
}

impl SearchInfo {
    pub fn new() -> Self {
        Self {
            start_time: None,
            nodes: 0,
            cp: 0,
            ply: 0,
            tt_hits: 0,
            killer_moves1: vec![None; 128],
            killer_moves2: vec![None; 128],
            capture_history: [[[0; 7]; 64]; 7],
            quiet_history: [[0; 64]; 7],
            counter_moves: vec![vec![None; 64]; 64],
            prev_move: vec![None; 128],
            terminated: false,
            pv: vec![vec![None; 128]; 128],
            pv_length: [0; 128],
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

    pub fn update_capture_history(&mut self, role: Role, to: Square, captured: Role, value: i16) {
        let clamped_value = value.clamp(-MAX_CAPTURE, MAX_CAPTURE);
        self.capture_history[role as usize][to as usize][captured as usize] += clamped_value
            - self.capture_history[role as usize][to as usize][captured as usize]
                * clamped_value.abs()
                / MAX_CAPTURE;
    }

    pub fn get_capture_score(&self, m: &Move) -> i16 {
        self.capture_history[m.role() as usize][m.to() as usize][m.capture().unwrap() as usize]
    }
    pub fn update_quiet_history(&mut self, role: Role, to: Square, value: i16) {
        let clamped_value = value.clamp(-MAX_HISTORY, MAX_HISTORY);
        self.quiet_history[role as usize][to as usize] += clamped_value
            - self.quiet_history[role as usize][to as usize] * clamped_value.abs() / MAX_HISTORY;
    }

    pub fn get_quiet_score(&self, m: &Move) -> i16 {
        self.quiet_history[m.role() as usize][m.to() as usize]
    }
}

pub struct SearchParams {
    pub depth: i8,
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
                if cnt > 2 {
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
        let hash = self.repetitions.last();
        match hash {
            Some(&h) => h,
            None => self.pos.zobrist_hash(EnPassantMode::Legal),
        }
    }
}
