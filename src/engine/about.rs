use super::Engine;

impl Engine {
    pub fn print_logo(&self) {
        println!();
        println!("▗▖ ▗▖▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖▗▖  ▗▖▗▄▄▄");
        println!("▐▌ ▐▌  █  ▐▌  ▐▌▐▌   ▐▛▚▞▜▌  █  ▐▛▚▖▐▌▐▌  █");
        println!("▐▛▀▜▌  █  ▐▌  ▐▌▐▛▀▀▘▐▌  ▐▌  █  ▐▌ ▝▜▌▐▌  █");
        println!("▐▌ ▐▌▗▄█▄▖ ▝▚▞▘ ▐▙▄▄▖▐▌  ▐▌▗▄█▄▖▐▌  ▐▌▐▙▄▄▀");
        println!();
    }

    pub fn print_options(&self) {
        println!("{:<10} {} {}", "Engine", "hivemind", "v1.0");
        println!("{:<10} {}", "Author", "aminwoo");
        println!();
        println!("option name UCI_Variant type combo default chess var chess var crazyhouse");
    }
}
