<div align="center">
  
  ![hivemind-logo](https://github.com/aminwoo/hivemind/assets/124148472/dd2c18ca-13f7-4cd3-9f6a-7928637465a9)
  
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
Weights for a network trained on 600k human games and further trained via self-play is provided. 

## Tournament Play 

## Cite This Work
If you found this work useful, please cite it with:
```
@software{hivemind,
  url = {https://github.com/aminwoo/hivemind}
}
```
