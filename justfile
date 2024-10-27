run:
  RUSTFLAGS="-C target-cpu=native" cargo run --release
build:
  RUSTFLAGS="-C target-cpu=native" cargo build --release
