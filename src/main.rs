mod bughouse;
mod engine;
mod search;

use engine::Engine;

fn main() {
    let mut engine = Engine::new();
    engine.run();
}
