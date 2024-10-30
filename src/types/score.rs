pub struct Score;

impl Score {
    pub const DRAW: i16 = 0;

    pub const INFINITY: i16 = 32000;

    pub const MATE: i16 = Self::INFINITY - 1000;
    pub const MATE_BOUND: i16 = Self::MATE - 500;

    pub const fn mated_in(ply: usize) -> i16 {
        -Self::MATE + ply as i16
    }
}
