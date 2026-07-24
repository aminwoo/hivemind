cmake --preset ninja-fast -DTensorRT_DIR=/home/ben/opt/TensorRT-11.1.0.106 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
cmake --build --preset ninja-fast -j "$(nproc)"
./build-ninja/hivemind

cmake --preset ninja-release -DTensorRT_DIR=/home/ben/opt/TensorRT-11.1.0.106 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
cmake --build --preset ninja-release -j "$(nproc)"
