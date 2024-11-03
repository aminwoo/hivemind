use crate::types::Score;
use shakmaty::{Chess, Color, Position};

const PIECE_VALUES: [i32; 7] = [0, 100, 320, 330, 500, 900, 20000];

pub fn evaluate<P: Position + Clone>(pos: &P) -> i32 {
    let board = pos.board();

    let score: i32 = board
        .occupied()
        .into_iter()
        .map(|square| {
            board.piece_at(square).map_or(0, |piece| {
                let role = piece.role as usize;
                let value = PIECE_VALUES[role];
                if piece.color == Color::White {
                    value
                } else {
                    -value
                }
            })
        })
        .sum();

    if pos.turn() == Color::Black {
        -score
    } else {
        score
    }
}

pub fn benchmark<P: Position + Clone>(
    pos: &P,
    mut alpha: i32,
    beta: i32,
    depth: i32,
    nodes: &mut i32,
) -> i32 {
    *nodes += 1;
    if depth <= 0 {
        return evaluate(pos);
    }

    let mut best_score = -Score::INFINITY;
    let moves = pos.legal_moves();

    for mv in moves {
        let mut child = pos.clone();
        child.play_unchecked(&mv);

        let score = -benchmark(&child, -beta, -alpha, depth - 1, nodes);

        if score > best_score {
            best_score = score;

            if score > alpha {
                alpha = score;
            }
        }

        if alpha >= beta {
            break;
        }
    }

    best_score
}

pub fn perft<P: Position + Clone>(pos: &P, depth: i32) -> u64 {
    if depth < 1 {
        1
    } else {
        let moves = pos.legal_moves();

        if depth == 1 {
            moves.len() as u64
        } else {
            let mut cnt = 0;
            for mv in moves {
                let mut child = pos.clone();
                child.play_unchecked(&mv);
                cnt += perft(&child, depth - 1);
            }
            cnt
        }
    }
}
