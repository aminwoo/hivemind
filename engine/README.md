cmake --preset ninja-fast -DTensorRT_DIR=/home/ben/opt/TensorRT-11.1.0.106 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
cmake --build --preset ninja-fast -j "$(nproc)"
./build-ninja/hivemind \
	--network "$(realpath ../src/training/weights/rl/model-rl-final-v3.0.onnx)"

## Asymmetric sit-boost tournament

Use the new network as the contender with its native sit logits, while retaining
the former artificial sit floors for the baseline:

```bash
NEW_NETWORK=/path/to/new-network.onnx
OLD_NETWORK=/path/to/old-network.onnx

./build-ninja/hivemind tournament \
	--contender "$NEW_NETWORK" \
	--baseline "$OLD_NETWORK" \
	--games 100 \
	--nodes 800 \
	--output tournament_results/sit-boost-ablation \
	--seed 1 \
	--contender-wait-pass-prior-floor 0 \
	--contender-coordination-pass-prior-floor 0 \
	--baseline-wait-pass-prior-floor 0.10 \
	--baseline-coordination-pass-prior-floor 0.05
```

The boost settings follow the network when colors are swapped in paired games
and are recorded in `summary.json`.

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

cmake --preset ninja-release -DTensorRT_DIR=/home/ben/opt/TensorRT-11.1.0.106 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
cmake --build --preset ninja-release -j "$(nproc)"
