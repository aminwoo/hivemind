#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --model MODEL.onnx [--output DIR] [--name NAME]" >&2
    echo "Builds a self-contained x86-64 Linux zip with Hivemind and TensorRT." >&2
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine_dir="$(cd "$script_dir/.." && pwd)"
workspace_dir="$(cd "$engine_dir/.." && pwd)"
model=""
output_dir="$workspace_dir/dist"
bundle_name="hivemind-v2.2.1-linux-x86_64-tensorrt"
tensorrt_dir="${TensorRT_DIR:-/home/ben/opt/TensorRT-11.1.0.106}"
cuda_dir="${CUDA_TOOLKIT_ROOT_DIR:-/usr/local/cuda}"

while (($#)); do
    case "$1" in
        --model) model="${2:?missing value for --model}"; shift 2 ;;
        --output) output_dir="${2:?missing value for --output}"; shift 2 ;;
        --name) bundle_name="${2:?missing value for --name}"; shift 2 ;;
        --tensorrt-dir) tensorrt_dir="${2:?missing value for --tensorrt-dir}"; shift 2 ;;
        --cuda-dir) cuda_dir="${2:?missing value for --cuda-dir}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$model" || ! -f "$model" ]]; then
    echo "A readable ONNX model is required; pass it with --model." >&2
    exit 2
fi
if [[ ! -d "$tensorrt_dir/lib" ]]; then
    echo "TensorRT library directory not found: $tensorrt_dir/lib" >&2
    exit 2
fi
command -v cmake >/dev/null
command -v zip >/dev/null

build_dir="$engine_dir/build-ubuntu-release"
cmake -S "$engine_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DHIVEMIND_FAST_BUILD=OFF \
    -DHIVEMIND_NATIVE_ARCH=OFF \
    -DTensorRT_DIR="$tensorrt_dir" \
    -DCUDA_TOOLKIT_ROOT_DIR="$cuda_dir"
cmake --build "$build_dir" --target hivemind -j "$(nproc)"

mkdir -p "$output_dir"
staging_root="$(mktemp -d "$output_dir/.hivemind-release.XXXXXX")"
trap 'rm -rf "$staging_root"' EXIT
bundle_dir="$staging_root/$bundle_name"
mkdir -p "$bundle_dir/bin" "$bundle_dir/lib" "$bundle_dir/models"

cp "$build_dir/hivemind.bin" "$bundle_dir/bin/"
cp "$model" "$bundle_dir/models/hivemind.onnx"
cp "$workspace_dir/LICENSE" "$bundle_dir/LICENSE.txt"
cp "$engine_dir/LICENSE" "$bundle_dir/FAIRY-STOCKFISH-LICENSE.txt"

for license in "$tensorrt_dir/LICENSE.txt" "$tensorrt_dir/LICENSE"; do
    if [[ -f "$license" ]]; then
        cp "$license" "$bundle_dir/TENSORRT-LICENSE.txt"
        break
    fi
done
for license in "$cuda_dir/LICENSE.txt" "$cuda_dir/LICENSE"; do
    if [[ -f "$license" ]]; then
        cp "$license" "$bundle_dir/CUDA-RUNTIME-LICENSE.txt"
        break
    fi
done

# The runtime and parser are direct dependencies. Builder resources are loaded
# dynamically when Hivemind creates a GPU-specific plan on first launch.
for library in \
    libnvinfer.so* \
    libnvonnxparser.so* \
    libnvinfer_builder_resource_sm*.so* \
    libnvinfer_builder_resource_ptx.so*; do
    for source in "$tensorrt_dir/lib"/$library; do
        [[ -e "$source" || -L "$source" ]] || continue
        cp -a "$source" "$bundle_dir/lib/"
    done
done

# CI uses NVIDIA's small CUDA Runtime redistributable rather than a full CUDA
# Toolkit. Bundle its dynamic runtime when present; locally, FindCUDA may have
# selected the static runtime, in which case this loop simply copies nothing.
for cuda_lib_dir in \
    "$cuda_dir/lib64" \
    "$cuda_dir/lib" \
    "$cuda_dir/targets/x86_64-linux/lib"; do
    [[ -d "$cuda_lib_dir" ]] || continue
    for source in "$cuda_lib_dir"/libcudart.so*; do
        [[ -e "$source" || -L "$source" ]] || continue
        cp -a "$source" "$bundle_dir/lib/"
    done
done

cp "$engine_dir/scripts/convert_onnx_fp16.py" "$bundle_dir/bin/"

cat >"$bundle_dir/hivemind" <<'LAUNCHER'
#!/bin/sh
set -eu
bundle_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export LD_LIBRARY_PATH="$bundle_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIVEMIND_TENSORRT_LIBRARY_DIR="$bundle_dir/lib"
export HIVEMIND_FP16_CONVERTER_SCRIPT="$bundle_dir/bin/convert_onnx_fp16.py"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Hivemind requires an NVIDIA driver (nvidia-smi was not found)." >&2
    exit 1
fi

for argument in "$@"; do
    case "$argument" in
        --model|--model=*|--network|--network=*)
            exec "$bundle_dir/bin/hivemind.bin" "$@"
            ;;
    esac
done
exec "$bundle_dir/bin/hivemind.bin" --model "$bundle_dir/models/hivemind.onnx" "$@"
LAUNCHER
chmod 755 "$bundle_dir/hivemind" "$bundle_dir/bin/hivemind.bin"

cat >"$bundle_dir/README.txt" <<'README'
Hivemind for Ubuntu x86-64
=========================

Requirements:
  - A 64-bit Ubuntu installation
  - A supported NVIDIA GPU and current proprietary NVIDIA driver
  - Enough free disk space for the GPU-specific TensorRT plan

Run:
  ./hivemind

The first launch builds and caches a TensorRT plan beside the bundled ONNX
model. This can take several minutes. Later launches reuse that plan. TensorRT
plans are GPU- and software-version-specific; distribute this ONNX-based bundle
instead of copying a plan generated on another computer.

Hivemind speaks UCI on stdin/stdout and can be configured as an engine in a
compatible GUI by selecting the top-level `hivemind` launcher.
README

zip_path="$output_dir/$bundle_name.zip"
rm -f "$zip_path"
(
    cd "$staging_root"
    zip -q -y -r "$zip_path" "$bundle_name"
)
digest="$(sha256sum "$zip_path" | cut -d ' ' -f1)"
printf '%s  %s\n' "$digest" "$(basename "$zip_path")" >"$zip_path.sha256"
echo "Created $zip_path"
echo "Created $zip_path.sha256"
