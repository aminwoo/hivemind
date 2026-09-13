## Inference backends

Download the [published network](https://huggingface.co/aminwoo/bughouse-rise-v3)
from the repository root with `python tools/fetch_network.py`. Files are verified
and installed into `engine/models`. See the [network setup instructions](../README.md#download-the-network)
for native FP16, CPU conversion, and training checkpoints. Pass `--model PATH`
to explicitly select the network for your backend.

The engine builds against either of two backends, selected with
`-DHIVEMIND_BACKEND=`:

|                      | `tensorrt` (default)                  | `onnxruntime`                   |
| -------------------- | ------------------------------------- | ------------------------------- |
| Requires             | CUDA 13+, TensorRT 10.14+, NVIDIA GPU | ONNX Runtime                    |
| Platforms            | Linux / Windows + NVIDIA              | Linux / macOS / Windows         |
| Precision            | FP16                                  | FP32 recommended; FP16 optional |
| Redistributable size | ~2 GB                                 | ~85 MB uncompressed             |

The TensorRT path is the default and is unchanged — existing build commands
behave exactly as before. The ONNX Runtime path is opt-in and exists so the
engine can be built and shipped without CUDA at all.

### Building the portable backend

```bash
python3 tools/fetch_onnxruntime.py          # ~11 MB, into third_party/
python3 -m pip install numpy onnx
python3 engine/scripts/convert_onnx_fp32.py model-fp16.onnx model-fp32.onnx
cmake -S engine -B engine/build-ort -G Ninja \
    -DHIVEMIND_BACKEND=onnxruntime \
    -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build-ort -j "$(nproc)"
```

CMake finds the runtime in `third_party/onnxruntime`; pass
`-DONNXRuntime_ROOT=<dir>` to use one installed elsewhere.

Load the converted FP32 network on both Windows and Linux:

```bash
./engine/build-ort/hivemind.bin --model model-fp32.onnx
```

An FP16-compatible build remains available by adding
`-DHIVEMIND_ORT_FP16=ON`, but FP16 execution is usually much slower on the CPU.

On Windows, run these commands from a Developer PowerShell for Visual Studio:

```powershell
py tools/fetch_onnxruntime.py
py -m pip install numpy onnx
py engine/scripts/convert_onnx_fp32.py model-fp16.onnx model-fp32.onnx
cmake -S engine -B engine/build-ort -A x64 `
  -DHIVEMIND_BACKEND=onnxruntime `
  -DONNXRuntime_ROOT=third_party/onnxruntime
cmake --build engine/build-ort --config Release --parallel
```

The executable and `onnxruntime.dll` are written to
`engine/build-ort/Release`. MSVC builds use native compiler options; no CUDA,
Unix shell, or POSIX compatibility layer is needed.

### Building the TensorRT backend

On Linux:

```bash
cmake --preset ninja-fast \
    -DTensorRT_DIR=/path/to/TensorRT -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
cmake --build --preset ninja-fast -j "$(nproc)"
./build-ninja/hivemind \
	--network "$(realpath ../artifacts/training/weights/rl/model-rl-final-v3.0.onnx)"
```

Self-play allocates a fixed Fairy-Stockfish mate-search budget per searched
position. The default 8,000,000 nodes are split across the active boards:

```bash
./build-ninja/hivemind selfplay \
  --network "$(realpath ../artifacts/training/weights/rl/model-rl-final-v3.0.onnx)" \
  --games 1000 --nodes 400 --fairy-stockfish-mate-nodes 8000000
```

Set `--fairy-stockfish-mate-nodes 0` to disable the probe.

### Internal mate-probe experiment

The `InternalMateProbe` UCI combo is disabled by default and separates the
experiment's effects:

| Mode        | Candidate telemetry | Selection bias | Exact solver update                      |
| ----------- | ------------------- | -------------- | ---------------------------------------- |
| `off`       | no                  | no             | no                                       |
| `telemetry` | yes                 | no             | no                                       |
| `bias`      | yes                 | yes            | no                                       |
| `certify`   | yes                 | yes            | only after exact two-board certification |

Use `setoption name InternalMateProbe value telemetry` to measure candidate
hit rates without changing play. That mode preserves 90% of the Fairy probe's
time for the established root probe. The strength-affecting `bias` and
`certify` modes cap that initial root phase at 10%, allowing internal hints to
arrive early enough to accumulate visits. All modes retain the shared node
budget, with 90% of its nodes reserved for the root probe in every mode. For an
untimed complete probe, the wall-clock cap does not apply, so the root probe
still runs first and may consume its full node share. A Fairy hit remains in
the hint table when exact certification fails; only successful certification
may update MCTS solver state.

On Windows, use a Developer PowerShell for Visual Studio and point CMake at
the extracted TensorRT SDK and CUDA Runtime redistributable:

```powershell
cmake -S engine -B engine/build-tensorrt -A x64 `
  -DHIVEMIND_BACKEND=tensorrt `
  -DTensorRT_DIR=C:\path\to\TensorRT-11.1.0.106 `
  -DCUDA_TOOLKIT_ROOT_DIR=C:\path\to\cuda-runtime
cmake --build engine/build-tensorrt --config Release --parallel
```

The Windows backend loads TensorRT builder-resource DLLs from the executable
directory. FP32-to-FP16 conversion uses `.venv\\Scripts\\python.exe` or
`python` and can be overridden with `HIVEMIND_PYTHON`.

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

## TensorRT release bundles

Build a generic x86-64 Linux ZIP containing the engine, an ONNX network, and
the local TensorRT and CUDA runtime libraries:

```bash
./scripts/package_ubuntu_release.sh \
    --model /path/to/model.onnx \
    --name hivemind-v2.2.2-linux-x86_64-tensorrt
```

Recipients need a supported NVIDIA GPU and proprietary NVIDIA driver, but do
not need to install CUDA or TensorRT. The first launch builds a TensorRT plan
for their GPU and caches it beside the bundled model.

Build the equivalent package from a Developer PowerShell on Windows:

```powershell
python engine/scripts/package_windows_tensorrt_release.py `
  --model C:\path\to\model.onnx `
  --tensorrt-root C:\path\to\TensorRT-11.1.0.106 `
  --cuda-root C:\path\to\cuda-runtime `
  --name hivemind-v2.2.2-windows-x86_64-tensorrt
```

## Portable Windows and Linux release bundles

Build the ONNX Runtime CPU bundle natively on either operating system:

```bash
python tools/fetch_onnxruntime.py
python -m pip install numpy onnx
python engine/scripts/package_portable_release.py \
    --model model-fp16.onnx --name hivemind-v2.2.2-linux-x86_64-onnxruntime
```

On Windows, use the same Python command and a
`hivemind-v2.2.2-windows-x86_64-onnxruntime` name. The ZIP includes the engine,
an automatically converted FP32 model, ONNX Runtime, licenses, and checksums.
It does not require a GPU.
Tagged builds create ONNX Runtime and TensorRT archives for both Linux and
Windows through `.github/workflows/portable-release.yml`.
