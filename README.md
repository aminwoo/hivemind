# HiveMind
Bughouse or doubles chess is a chess variant involving two teams and two chess boards playing in parallel. 

This repo features 
* Utilises for downloading and parsing bughouse games from chess.com and fics
* Training scripts for both supervised learning and reinforcement learning using gumbel alphazero
* A client to play and test out the network

## Installing Miniconda

```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

conda create -n py39 python=3.9
conda activate py39
```

## Installation

`hivemind` uses JAX, which can be installed with 
```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
```
pip install tensorflow[and-cuda]
```
The rest of the dependencies are handled by `poetry` and is as simple as - 
```
pip install poetry
poetry install 
```
to install. 

## Installing pgx and mctz 

```
pip install git+https://github.com/aminwoo/pgx.git
pip install git+https://github.com/lowrollr/mctx-az.git
```

## Playing against the Engine

## Training


## Cite This Work
If you found this work useful, please cite it with:
```
@software{hivemind,
  url = {https://github.com/aminwoo/hivemind}
}
```
