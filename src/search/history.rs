use crate::types::parameters::*;
use shakmaty::{Chess, Color, Move, MoveList, Position, Role, Square};

const MAX_HISTORY: i32 = 16384;

type Butterfly<T> = [[T; 64]; 64];
type PieceSquare<T> = [[T; 64]; 8];

#[derive(Clone)]
pub struct History {
    capture: Box<[Butterfly<[i32; 8]>; 2]>,
    main: Box<[Butterfly<i32>; 2]>,
}

impl History {
    pub fn get_capture(&self, side: Color, mv: Move) -> Option<i32> {
        Some(
            self.capture[side as usize][mv.from()? as usize][mv.to() as usize]
                [mv.capture()? as usize],
        )
    }

    pub fn get_main(&self, stm: Color, mv: Move) -> Option<i32> {
        Some(self.main[stm as usize][mv.from()? as usize][mv.to() as usize])
    }

    pub fn update_capture(
        &mut self,
        pos: Chess,
        mv: Move,
        fails: MoveList,
        depth: i32,
    ) -> Option<()> {
        let turn = pos.turn() as usize;
        increase(
            &mut self.capture[turn][mv.from()? as usize][mv.to() as usize][mv.capture()? as usize],
            depth,
        );
        for fail in fails {
            decrease(
                &mut self.capture[turn][fail.from()? as usize][fail.to() as usize]
                    [fail.capture()? as usize],
                depth,
            );
        }
        Some(())
    }

    pub fn update_main(&mut self, stm: Color, mv: Move, fails: MoveList, depth: i32) -> Option<()> {
        increase(
            &mut self.main[stm as usize][mv.from()? as usize][mv.to() as usize],
            depth,
        );
        for fail in fails {
            decrease(
                &mut self.main[stm as usize][fail.from()? as usize][fail.to() as usize],
                depth,
            );
        }
        Some(())
    }
}

impl Default for History {
    fn default() -> Self {
        Self {
            main: zeroed_box(),
            capture: zeroed_box(),
        }
    }
}

fn bonus(depth: i32) -> i32 {
    (history_bonus() * depth + history_bonus_base()).min(history_bonus_max())
}

fn malus(depth: i32) -> i32 {
    (history_malus() * depth + history_malus_base()).min(history_malus_max())
}

fn increase(v: &mut i32, depth: i32) {
    let bonus = bonus(depth);
    *v += bonus - bonus * *v / MAX_HISTORY;
}

fn decrease(v: &mut i32, depth: i32) {
    let malus = malus(depth);
    *v -= malus + malus * *v / MAX_HISTORY;
}

fn zeroed_box<T>() -> Box<T> {
    unsafe {
        let layout = std::alloc::Layout::new::<T>();
        let ptr = std::alloc::alloc_zeroed(layout);
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Box::<T>::from_raw(ptr.cast())
    }
}
