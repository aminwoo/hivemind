use super::Search;
use shakmaty::MoveList;

pub const MVV_LVA: [[u8; 7]; 7] = [
    [0, 0, 0, 0, 0, 0, 0],       // victim None, attacker None, P, N, B, R, Q, K
    [0, 15, 14, 13, 12, 11, 10], // victim P, attacker None, P, N, B, R, Q, K
    [0, 25, 24, 23, 22, 21, 20], // victim N, attacker None, P, N, B, R, Q, K
    [0, 35, 34, 33, 32, 31, 30], // victim B, attacker None, P, N, B, R, Q, K
    [0, 45, 44, 43, 42, 41, 40], // victim R, attacker None, P, N, B, R, Q, K
    [0, 55, 54, 53, 52, 51, 50], // victim Q, attacker None, P, N, B, R, Q, K
    [0, 0, 0, 0, 0, 0, 0],       // victim K, attacker None, P, N, B, R, Q, K
];

impl Search {
    pub fn sort_moves(moves: &mut MoveList) {
        moves.sort_by_key(|mv| {
            let piece = mv.role() as usize;
            let captured = match mv.capture() {
                Some(role) => role as usize,
                None => 0,
            };
            -(MVV_LVA[captured][piece] as isize)
        });
    }
}
