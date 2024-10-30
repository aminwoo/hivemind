mod bughouse;
mod engine;
mod search;

use bughouse::Bughouse;
use engine::Engine;
use rand::seq::SliceRandom;
use rand::thread_rng;

use shakmaty::{fen::Fen, Bitboard, CastlingMode, Chess, Move, Position, Square};

pub const SEE_VALUE: [i16; 7] = [0, 100, 325, 325, 500, 1000, 30000];

fn see(pos: &Chess, m: &Move) -> i16 {
    let board = pos.board();

    let mut gain: [i16; 32] = [0; 32];
    let mut depth = 0;

    let mut from_set = Bitboard::from(m.from().unwrap());
    let mut occupied = board.occupied();

    let mut target = 0;
    let mut attacker = m.role() as usize;
    if let Some(role) = m.capture() {
        target = role as usize;
    }
    gain[depth] = SEE_VALUE[target];

    let mut attackers;
    loop {
        depth += 1;
        gain[depth] = SEE_VALUE[attacker] - gain[depth - 1];
        occupied ^= from_set;

        if depth & 1 == 1 {
            attackers = board.attacks_to(m.to(), pos.turn().other(), occupied) & occupied;
        } else {
            attackers = board.attacks_to(m.to(), pos.turn(), occupied) & occupied;
        }

        from_set = attackers & board.pawns();
        if from_set.any() {
            from_set = from_set.isolate_first();
            attacker = 1;
            continue;
        }
        from_set = attackers & board.knights();
        if from_set.any() {
            from_set = from_set.isolate_first();
            attacker = 2;
            continue;
        }
        from_set = attackers & board.bishops();
        if from_set.any() {
            from_set = from_set.isolate_first();
            attacker = 3;
            continue;
        }
        from_set = attackers & board.rooks();
        if from_set.any() {
            from_set = from_set.isolate_first();
            attacker = 4;
            continue;
        }
        from_set = attackers & board.queens();
        if from_set.any() {
            from_set = from_set.isolate_first();
            attacker = 5;
            continue;
        }
        break;
    }

    depth -= 1;
    while depth > 0 {
        gain[depth - 1] = -std::cmp::max(-gain[depth - 1], gain[depth]);
        depth -= 1;
    }
    gain[0]
}

fn main() {
    let mut engine = Engine::new();
    engine.run();
    /*let fen_string = String::from("1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - -");
    let fen: Fen = fen_string.parse().unwrap();
    let mut pos: Chess = fen.into_position(CastlingMode::Standard).unwrap();
    let legal_moves = pos.legal_moves();
    for m in legal_moves {
        println!("{}", m);
        println!("{}", see(&pos, &m));
    }*/
    /*let mut pos = Bughouse::new();


    let mut rng = thread_rng();
    loop {
        let legal_moves = pos.legal_moves();
        let m_option = legal_moves.choose(&mut rng);
        if let Some(m) = m_option {
            println!("{:?}", m);
            pos.play_unchecked(m);
        }
        if pos.is_checkmate() {
            break;
        }
    }*/
}
