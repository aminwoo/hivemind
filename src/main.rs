mod bughouse;
mod engine;
mod search;
mod transposition;
mod types;

use engine::Engine;

fn main() {
    let mut engine = Engine::new();
    engine.run();
}
