use super::defs::SearchRefs;
use super::{Move, Search};
use crate::types::parameters::*;
use shakmaty::{
    attacks::{bishop_attacks, rook_attacks},
    Bitboard, Board, Chess, Color, MoveList, Position, Role,
};

const BAD_CAPTURE: i32 = -200_000_000;
const GOOD_CAPTURE: i32 = 200_000_000;
const KILLER_BONUS: i32 = 100_000_000;
const HASH_MOVE: i32 = 300_000_000;

pub const SEE_VALUE: [i32; 7] = [0, 100, 400, 400, 650, 1200, 0];

pub fn least_valuable_attacker(board: &Board, attackers: Bitboard) -> Option<Role> {
    Role::ALL
        .into_iter()
        .find(|&attacker| (attackers & board.by_role(attacker)).any())
}

pub fn see(pos: &Chess, mv: &Move, threshold: i32) -> Option<bool> {
    if mv.is_promotion() || mv.is_castle() {
        return Some(true);
    }

    let board = pos.board();
    let mut balance = SEE_VALUE[mv.capture()? as usize] - threshold;
    if balance < 0 {
        return Some(false);
    }

    balance -= SEE_VALUE[mv.role() as usize];
    if balance >= 0 {
        return Some(true);
    }

    let mut occupied = board.occupied();
    occupied.remove(mv.from()?);
    occupied.set(mv.to(), true);

    let mut stm = pos.turn().other();
    let mut attackers = (board.attacks_to(mv.to(), Color::White, occupied)
        | board.attacks_to(mv.to(), Color::Black, occupied))
        & occupied;

    let diagonal = board.bishops() | board.queens();
    let orthogonal = board.rooks() | board.queens();

    loop {
        let our_attackers = attackers & board.by_color(stm);
        if our_attackers.is_empty() {
            break;
        }
        let attacker =
            least_valuable_attacker(board, our_attackers).expect("Expected at least 1 attacker");
        if attacker == Role::King && (attackers & board.by_color(stm.other())).any() {
            break;
        }

        occupied ^= (board.by_role(attacker) & our_attackers).isolate_first();
        stm = stm.other();

        balance = -balance - 1 - SEE_VALUE[attacker as usize];
        if balance >= 0 {
            break;
        }

        if [Role::Pawn, Role::Bishop, Role::Queen].contains(&attacker) {
            attackers |= bishop_attacks(mv.to(), occupied) & diagonal;
        }
        if [Role::Rook, Role::Queen].contains(&attacker) {
            attackers |= rook_attacks(mv.to(), occupied) & orthogonal;
        }
        attackers &= occupied;
    }

    Some(stm != pos.turn())
}

impl Search {
    pub fn sort_moves(
        moves: &mut MoveList,
        pv_move: &Option<Move>,
        tt_move: &Option<Move>,
        refs: &SearchRefs,
    ) {
        moves.sort_by_key(|m| {
            if let Some(mv) = &pv_move {
                if mv == m {}
            }
            if let Some(mv) = &tt_move {
                if mv == m {
                    return HASH_MOVE;
                }
            }

            if m.is_capture() {
                let captured = match m.capture() {
                    Some(role) => role as usize,
                    None => 0,
                };
                let see_value = see(refs.pos, m, 0).expect("Error calculating SEE");
                let history = refs
                    .search_info
                    .history
                    .get_capture(refs.pos.turn(), m.clone())
                    .expect("Expected move to be a capture");
                let mvv = 32 * SEE_VALUE[captured];
                if !see_value {
                    return BAD_CAPTURE + history + mvv;
                }
                return GOOD_CAPTURE + history + mvv;
            }
            let ply = refs.search_info.ply as usize;
            if let Some(killer) = &refs.search_info.killers[ply] {
                if m == killer {
                    return KILLER_BONUS;
                }
            }
            ordering_main()
                * refs
                    .search_info
                    .history
                    .get_main(refs.pos.turn(), m.clone())
                    .expect("Error getting FROM square")
        });
        moves.reverse();
    }
}

#[cfg(test)]
mod tests {
    use crate::search::sorting::see;
    use shakmaty::fen::Fen;
    use shakmaty::uci::UciMove;
    use shakmaty::{CastlingMode, Chess};
    #[test]
    fn test_see1() {
        let fen_string = String::from("1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - ");
        let fen: Fen = fen_string.parse().unwrap();
        let pos: Chess = fen.into_position(CastlingMode::Standard).unwrap();

        let uci: UciMove = "d3e5".parse().unwrap();
        let mv = uci.to_move(&pos).unwrap();

        assert_eq!(see(&pos, &mv, -225), Some(true));
    }
    #[test]
    fn test_see2() {
        let fen_string = String::from("1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - ");
        let fen: Fen = fen_string.parse().unwrap();
        let pos: Chess = fen.into_position(CastlingMode::Standard).unwrap();

        let uci: UciMove = "e1e5".parse().unwrap();
        let mv = uci.to_move(&pos).unwrap();

        assert_eq!(see(&pos, &mv, 101), Some(false));
    }
}
