MODEL  := "v5-0aa6d222.nnue"
REPO   := "https://github.com/codedeliveryservice/RecklessNetworks/raw/main"

run:
  RUSTFLAGS="-C target-cpu=native" cargo run --release
build:
  RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release
test:
  cargo test
fetch: 
  echo Downloading {{MODEL}}
  curl -sL {{REPO}}/{{MODEL}} -o networks/model.nnue --create-dirs

