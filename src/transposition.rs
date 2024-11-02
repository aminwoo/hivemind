use crate::types::Score;

use shakmaty::Move;
pub const DEFAULT_TT_SIZE: usize = 1024;

const MEGABYTE: usize = 1024 * 1024;
const INTERNAL_ENTRY_SIZE: usize = std::mem::size_of::<InternalEntry>();

#[derive(Clone)]
pub struct Entry {
    pub m: Option<Move>,
    pub score: i32,
    pub depth: i32,
    pub bound: Bound,
}

impl Entry {
    pub const fn valid_cutoff(&self, alpha: i32, beta: i32, depth: i32) -> bool {
        match self.bound {
            _ if depth > self.depth => false,
            Bound::Exact => true,
            Bound::Beta => self.score >= beta,
            Bound::Alpha => self.score <= alpha,
            Bound::Nothing => false,
        }
    }
}

/// Type of the score returned by the search.
#[derive(Copy, Clone, PartialEq)]
pub enum Bound {
    Exact,
    Alpha,
    Beta,
    Nothing,
}

/// Internal representation of a transposition table entry (8 bytes).
#[derive(Clone)]
struct InternalEntry {
    key: u16,
    m: Option<Move>,
    score: i32,
    depth: i32,
    bound: Bound,
    valid: bool,
}

impl Default for InternalEntry {
    fn default() -> Self {
        Self {
            key: 0,
            m: None,
            score: 0,
            depth: 0,
            bound: Bound::Nothing,
            valid: false,
        }
    }
}

/// The transposition table is used to cache previously performed search results.
pub struct TranspositionTable {
    vector: Vec<InternalEntry>,
}

impl TranspositionTable {
    /// Clears the transposition table. This will remove all entries but keep the allocated memory.
    pub fn clear(&mut self) {
        self.vector.fill(InternalEntry::default());
    }

    /// Resizes the transposition table to the specified size in megabytes. This will clear all entries.
    pub fn resize(&mut self, megabytes: usize) {
        let len = megabytes * MEGABYTE / INTERNAL_ENTRY_SIZE;

        self.vector = Vec::new();
        self.vector.reserve_exact(len);

        unsafe {
            self.vector.set_len(len);
        }
        self.clear();
    }

    /// Returns the approximate load factor of the transposition table in permille (on a scale of `0` to `1000`).
    pub fn hashfull(&self) -> usize {
        self.vector
            .iter()
            .take(1000)
            .filter(|slot| slot.valid)
            .count()
    }

    pub fn read(&self, hash: u64, ply: usize) -> Option<Entry> {
        let index = self.index(hash);
        let entry = self.vector[index].clone();
        if !entry.valid || entry.key != verification_key(hash) {
            return None;
        }

        let mut hit = Entry {
            m: entry.m,
            depth: entry.depth,
            score: entry.score,
            bound: entry.bound,
        };
        // Adjust mate distance from "plies from the current position" to "plies from the root"
        if hit.score.abs() > Score::MATE_BOUND {
            hit.score -= hit.score.signum() * ply as i32;
        }
        Some(hit)
    }

    pub fn write(
        &mut self,
        hash: u64,
        depth: i32,
        mut score: i32,
        bound: Bound,
        mut m: Option<Move>,
        ply: usize,
    ) {
        // Adjust mate distance from "plies from the root" to "plies from the current position"
        if score.abs() > Score::MATE_BOUND {
            score += score.signum() * ply as i32;
        }

        let key = verification_key(hash);
        let index = self.index(hash);

        let entry = self.vector[index].clone();
        if m.is_none() && entry.key == key {
            if let Some(old_m) = entry.m {
                m = Some(old_m);
            }
        }

        self.vector[index] = InternalEntry {
            key,
            m,
            depth,
            score,
            bound,
            valid: true,
        };
    }

    pub fn prefetch(&self, hash: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

            let index = self.index(hash);
            let ptr = self.vector.as_ptr().add(index).cast();
            _mm_prefetch::<_MM_HINT_T0>(ptr);
        }
    }

    fn index(&self, hash: u64) -> usize {
        // Fast hash table index calculation
        // For details, see: https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction
        (((hash as u128) * (self.vector.len() as u128)) >> 64) as usize
    }
}

/// Returns the verification key of the hash (bottom 16 bits).
const fn verification_key(hash: u64) -> u16 {
    hash as u16
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self {
            vector: vec![
                InternalEntry::default();
                DEFAULT_TT_SIZE * MEGABYTE / INTERNAL_ENTRY_SIZE
            ],
        }
    }
}
