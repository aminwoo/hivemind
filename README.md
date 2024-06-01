<div align="center">

  ![hivemind-logo](https://github.com/aminwoo/hivemind/assets/124148472/eede1e40-e215-403b-af97-c75e6adf7db0)
  
  <h3>Hivemind</h3>

  A free and strong UCI bughouse engine.

</div>

## Overview
Bughouse is a chess variant where two teams of two play against each other on two chess boards. Each team sits next to each other and the boards are arranged such that one player has white and the other has black. 

Once the game starts the boards play in parallel and captured pieces are exchanged in the teams - which can be dropped on empty squares like in shogi. These added complexities can result in mayhem, with each board frantically trying to deliver checkmate. 

This project was born out of a desire to provide a method of analysis to the bughouse community and advance the knowledge of this relatively unstudied game.

This repo features 
* Utilises for downloading and parsing bughouse games from chess.com and fics
* A modified architecture of the one presented in the AlphaZero paper which includes more recent advancements 
* Training scripts which supports both supervised learning and reinforcement learning using gumbel alphazero
* A client to play and test out the network

## Installation

`Hivemind` uses JAX, which can be installed with 
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

```
pip install git+https://github.com/aminwoo/pgx.git
pip install git+https://github.com/lowrollr/mctx-az.git
```

## Training
Weights for a network trained on 600k human games and further trained via self-play is provided. A trainer interface is provided to train your own model from scratch of continue training existing checkpoints. The module expects a policy target which sums to 1 and a value target (-1, 0 or 1). In the case of supervised learning, this will be a one-hot encoding of the expected action. Otherwise, it will be the policy action weights generated from the alphazero algorithm. 

## Tournament Play 

## Contributing

If you are interested in contributing to this project you can do so in several ways. 

1. Donating GPU compute through Kaggle (30 hours free GPU compute)
2. Improving documentation and instructions for this repo
3. Suggesting bug fixes and improvements to the engine

## Cite This Work
If you found this work useful, please cite it with:
```
@software{hivemind,
  url = {https://github.com/aminwoo/hivemind}
}
```
