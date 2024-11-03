use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    generate_model_env();

    println!("cargo:rerun-if-env-changed=EVALFILE");
    println!("cargo:rerun-if-changed=networks/model.nnue");
}

fn generate_model_env() {
    let mut path = env::var("EVALFILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new("networks").join("model.nnue"));

    if path.is_relative() {
        path = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    }

    println!("cargo:rustc-env=MODEL={}", path.display());
}
