mod benchmark;
mod board;
mod bughouse;
mod engine;
mod nnue;
mod search;
mod transposition;
mod types;

use engine::Engine;

fn main() {
    let mut engine = Engine::new();
    engine.run();
}
