run:
  RUSTFLAGS="-C target-cpu=native" cargo run --release
build:
  RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release
test:
  cargo test
