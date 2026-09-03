## Inference backends

The engine builds against either of two backends, selected with
`-DHIVEMIND_BACKEND=`:

| | `tensorrt` (default) | `onnxruntime` |
|---|---|---|
| Requires | CUDA 13+, TensorRT 10.14+, NVIDIA GPU | ONNX Runtime |
| Platforms | Linux + NVIDIA | Linux / macOS / Windows |
| Precision | FP16 | FP32 |
| Redistributable size | ~2 GB | ~28 MB |

The TensorRT path is the default and is unchanged — existing build commands
behave exactly as before. The ONNX Runtime path is opt-in and exists so the
engine can be built and shipped without CUDA at all.

### Building the portable backend

```bash
python3 tools/fetch_onnxruntime.py          # ~11 MB, into third_party/
cmake -S engine -B engine/build-ort -G Ninja \
    -DHIVEMIND_BACKEND=onnxruntime -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build-ort -j "$(nproc)"
```

CMake finds the runtime in `third_party/onnxruntime`; pass
`-DONNXRuntime_ROOT=<dir>` to use one installed elsewhere.

**Convert the network to FP32 first.** ONNX Runtime's CPU provider has no
native FP16 kernels and falls back to per-node casts, which is orders of
magnitude slower — slow enough to look like a hang. The engine refuses an FP16
network on this backend and points at the converter:

```bash
python3 engine/scripts/convert_onnx_fp32.py models/hivemind.onnx models/hivemind-fp32.onnx
./engine/build-ort/hivemind.bin --model models/hivemind-fp32.onnx
```

The file roughly doubles (27 MB -> 54 MB); that is the cost of portability.

### Building the TensorRT backend

Unchanged from before:

```bash
cmake --preset ninja-fast \
    -DTensorRT_DIR=/path/to/TensorRT -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

cmake --preset ninja-fast -DTensorRT_DIR=/home/ben/opt/TensorRT-11.1.0.106 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
cmake --build --preset ninja-fast -j "$(nproc)"
./build-ninja/hivemind \
	--network "$(realpath ../src/training/weights/rl/model-rl-final-v3.0.onnx)"

When a TensorRT plan is missing or stale, Hivemind accepts an FP32 ONNX model
and converts it to a temporary all-FP16 graph before building the plan. Plan
generation requires `onnx` and `onnxruntime` in `../.venv` or `python3`; set
`HIVEMIND_PYTHON` to select another Python interpreter. The cached TensorRT
plan uses FP16 inputs, outputs, weights, and floating-point activations.

## Progressive-widening tournament

Use the same network on both sides to isolate different progressive-widening
coefficients. Each value controls widening at both root and non-root nodes, and
the settings follow contender and baseline when colors swap:

```bash
NETWORK=/path/to/network.onnx

./build-ninja/hivemind tournament \
	--contender "$NETWORK" \
	--baseline "$NETWORK" \
	--games 100 \
	--nodes 800 \
	--output tournament_results/pw-1.5-vs-1.0 \
	--seed 1 \
	--contender-pw-coefficient 1.5 \
	--baseline-pw-coefficient 1.0
```

Both coefficients are recorded in `summary.json`.

For a fixed-time batch-size comparison, use the same network on both sides:

```bash
./build-ninja/hivemind tournament \
	--contender "$NETWORK" \
	--baseline "$NETWORK" \
	--games 20 \
	--movetime 1000 \
	--contender-batch-size 8 \
	--baseline-batch-size 16 \
	--output tournament_results/batch-8-vs-16 \
	--seed 1
```

For multi-parameter strength sweeps, the resumable runner forms the Cartesian
product of repeated `--axis` values. It supports batch size, worker count,
MCGS/transpositions, root mate search, progressive widening, WDL weight,
moves-left discount, and Q selection:

```bash
./scripts/run_strength_sweep.py \
    --engine ./build-ninja/hivemind.bin \
    --model "$NETWORK" \
    --games 400 --nodes 800 \
    --axis batch-size=8,16,32 \
    --axis threads=1,2,4 \
    --sprt-elo0 0 --sprt-elo1 8 \
    --positions tournament_positions.tsv \
    --output tournament_results/batch-workers --resume
```

The optional positions file is tab-separated: `dual FEN`, `white|black` team
to play, and `true|false` time advantage. Comment lines begin with `#`. Each
position is used for a complete color-swapped pair. Sequential stopping is
evaluated only after both games in a pair; `summary.json` also records measured
full-search NPS for each contestant and every effective search parameter.

cmake --preset ninja-release -DTensorRT_DIR=/home/ben/opt/TensorRT-11.1.0.106 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
cmake --build --preset ninja-release -j "$(nproc)"

## Ubuntu release bundle

Build a generic x86-64 Ubuntu zip containing the engine, an ONNX network, and
the local TensorRT runtime and builder libraries:

```bash
./scripts/package_ubuntu_release.sh \
    --model /path/to/model.onnx \
    --name hivemind-v2.1.0-ubuntu-x86_64
```

Recipients need a supported NVIDIA GPU and proprietary NVIDIA driver, but do
not need to install CUDA or TensorRT. The first launch builds a TensorRT plan
for their GPU and caches it beside the bundled model.
