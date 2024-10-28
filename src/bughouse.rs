mod defs;

use shakmaty::variant::Crazyhouse;
use shakmaty::{Color, Move, Position};

#[derive(Clone)]
pub struct Bughouse {
    left: Crazyhouse,
    right: Crazyhouse,
    turn: Color, // Turn of left board
}

#[derive(Debug)]
pub enum BoardSide {
    Left,
    Right,
}

impl Bughouse {
    pub const fn new() -> Bughouse {
        Bughouse {
            left: Crazyhouse::new(),
            right: Crazyhouse::new(),
            turn: Color::White,
        }
    }

    pub fn legal_moves(&self) -> Vec<(Move, BoardSide)> {
        let mut moves: Vec<(Move, BoardSide)> = Vec::new();
        if self.left.turn() == self.turn {
            moves.extend(
                self.left
                    .legal_moves()
                    .iter()
                    .cloned()
                    .map(|mv| (mv, BoardSide::Left)),
            );
        }
        if self.right.turn() != self.turn {
            moves.extend(
                self.right
                    .legal_moves()
                    .iter()
                    .cloned()
                    .map(|mv| (mv, BoardSide::Right)),
            );
        }

        moves
    }

    pub fn play_unchecked(&mut self, m: &(Move, BoardSide)) {
        match m.1 {
            BoardSide::Left => self.left.play_unchecked(&m.0),
            BoardSide::Right => self.right.play_unchecked(&m.0),
        }
        self.turn = self.turn.other();
    }

    pub fn is_check(&self) -> bool {
        (self.left.turn() == self.turn && self.left.is_check())
            || (self.right.turn() != self.turn && self.right.is_check())
    }

    pub fn is_checkmate(&self) -> bool {
        self.left.is_checkmate() || self.right.is_checkmate()
    }
}
