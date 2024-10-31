use shakmaty::Move;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::types::Score;

pub const DEFAULT_TT_SIZE: usize = 16;

const MEGABYTE: usize = 1024 * 1024;
const INTERNAL_ENTRY_SIZE: usize = std::mem::size_of::<InternalEntry>();

#[derive(Copy, Clone)]
pub struct Entry {
    pub score: i16,
    pub depth: i8,
    pub bound: Bound,
}

/// Type of the score returned by the search.
#[derive(Copy, Clone, PartialEq)]
pub enum Bound {
    Exact,
    Alpha,
    Beta,
}

/// Internal representation of a transposition table entry (8 bytes).
struct InternalEntry {
    key: u16,     // 2 bytes
    key2: u16,    // 2 bytes
    score: i16,   // 2 bytes
    depth: i8,    // 1 byte
    bound: Bound, // 1 byte
}

#[derive(Default)]
struct Block(AtomicU64);

impl Block {
    fn load(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    fn read(&self) -> Option<InternalEntry> {
        match self.load() {
            0 => None,
            v => Some(unsafe { std::mem::transmute::<u64, InternalEntry>(v) }),
        }
    }

    fn write(&self, entry: InternalEntry) {
        self.0.store(
            unsafe { std::mem::transmute::<InternalEntry, u64>(entry) },
            Ordering::Relaxed,
        );
    }
}

impl Clone for Block {
    fn clone(&self) -> Self {
        Self(AtomicU64::new(self.load()))
    }
}

/// The transposition table is used to cache previously performed search results.
pub struct TranspositionTable {
    vector: Vec<Block>,
}

impl TranspositionTable {
    /// Clears the transposition table. This will remove all entries but keep the allocated memory.
    pub fn clear(&mut self, threads: usize) {
        unsafe { self.parallel_clear(threads, self.vector.len()) }
    }

    /// Resizes the transposition table to the specified size in megabytes. This will clear all entries.
    pub fn resize(&mut self, threads: usize, megabytes: usize) {
        let len = megabytes * MEGABYTE / INTERNAL_ENTRY_SIZE;

        self.vector = Vec::new();
        self.vector.reserve_exact(len);

        unsafe {
            self.parallel_clear(threads, len);
            self.vector.set_len(len);
        }
    }

    /// Returns the approximate load factor of the transposition table in permille (on a scale of `0` to `1000`).
    pub fn hashfull(&self) -> usize {
        self.vector
            .iter()
            .take(1000)
            .filter(|slot| slot.load() != 0)
            .count()
    }

    pub fn read(&self, hash: u64, ply: usize) -> Option<Entry> {
        let entry = match self.entry(hash).read() {
            Some(v) if v.key == verification_key(hash) => v,
            _ => return None,
        };

        let mut hit = Entry {
            depth: entry.depth,
            score: entry.score,
            bound: entry.bound,
        };

        // Adjust mate distance from "plies from the current position" to "plies from the root"
        if hit.score.abs() > Score::MATE_BOUND {
            hit.score -= hit.score.signum() * ply as i16;
        }
        Some(hit)
    }

    pub fn write(&self, hash: u64, depth: i8, mut score: i16, bound: Bound, ply: usize) {
        // Adjust mate distance from "plies from the root" to "plies from the current position"
        if score.abs() > Score::MATE_BOUND {
            score += score.signum() * ply as i16;
        }

        let key = verification_key(hash);
        let entry = self.entry(hash);

        entry.write(InternalEntry {
            key,
            key2: key,
            depth: depth as i8,
            score: score as i16,
            bound,
        });
    }

    pub fn prefetch(&self, hash: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

            let index = self.index(hash);
            let ptr = self.vector.as_ptr().add(index).cast();
            _mm_prefetch::<_MM_HINT_T0>(ptr);
        }

        // No prefetching for non-x86_64 architectures
        #[cfg(not(target_arch = "x86_64"))]
        let _ = hash;
    }

    fn entry(&self, hash: u64) -> &Block {
        let index = self.index(hash);
        unsafe { self.vector.get_unchecked(index) }
    }

    fn index(&self, hash: u64) -> usize {
        // Fast hash table index calculation
        // For details, see: https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction
        (((hash as u128) * (self.vector.len() as u128)) >> 64) as usize
    }

    unsafe fn parallel_clear(&mut self, threads: usize, len: usize) {
        std::thread::scope(|scope| {
            let ptr = self.vector.as_mut_ptr() as *mut std::mem::MaybeUninit<Block>;
            let slice = std::slice::from_raw_parts_mut(ptr, len);

            let chunk_size = (len + threads - 1) / threads;
            for chunk in slice.chunks_mut(chunk_size) {
                scope.spawn(|| chunk.as_mut_ptr().write_bytes(0, chunk.len()));
            }
        });
    }
}

/// Returns the verification key of the hash (bottom 16 bits).
const fn verification_key(hash: u64) -> u16 {
    hash as u16
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self {
            vector: vec![Block::default(); DEFAULT_TT_SIZE * MEGABYTE / INTERNAL_ENTRY_SIZE],
        }
    }
}
