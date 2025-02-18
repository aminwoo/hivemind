<div align="center">
  
  ![hivemind-logo](https://github.com/aminwoo/hivemind/assets/124148472/d42c6a6e-ab2e-4d7a-bf90-4876d59c9558)
  
  <h3>HiveMind</h3>

  A free & strong UCI Bughouse engine.

</div>

## Overview
Bughouse is a four player chess variant where players exchange captured pieces which can be dropped on empty squares. The large game tree and complexities around coordination between players makes Bughouse a challenging domain for traditional Chess algorithms. HiveMind is a two board engine which uses a policy network and Monte Carlo tree search (MCTS) to narrow down the search space and coordinate both boards. 

## Prerequisites

This project requires the installation of the CUDA Toolkit and TensorRT for inference.

---

### CUDA Toolkit Installation Steps

Follow these steps to install the CUDA Toolkit:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cuda-toolkit-12-8
```

### TensorRT Installation Steps 
```
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.8.0/local_repo/nv-tensorrt-local-repo-ubuntu2204-10.8.0-cuda-12.8_1.0-1_amd64.deb
os="ubuntu2204"
tag="10.8.0-cuda-12.8"
dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get install tensorrt
```

## Compiling and Running HiveMind

```
cmake .
make
./hivemind
```


## Acknowledgements

*   [**CrazyAra**](https://github.com/QueensGambit/CrazyAra/tree/master): A Deep Learning UCI-Chess Variant Engine written in C++ & Python
*   [**Fairy-Stockfish**](https://github.com/fairy-stockfish/Fairy-Stockfish): A chess variant engine supporting Xiangqi, Shogi, Janggi, Makruk, S-Chess, Crazyhouse, Bughouse, and many more

## Licence

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](https://github.com/aminwoo/hivemind/blob/master/LICENSE) file for the full license text.
