use shakmaty::Move;

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct BughouseMove {
    data: Move,
    board: i8,
}
