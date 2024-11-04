<div align="center">
  
  ![hivemind-logo](https://github.com/aminwoo/hivemind/assets/124148472/d42c6a6e-ab2e-4d7a-bf90-4876d59c9558)
  
  <h3>HiveMind</h3>

  A free UCI Crazyhouse engine derived from Reckless and Rustic.

</div>

## Overview
Crazyhouse is a chess variant in which captured enemy pieces can be reintroduced, or dropped, into the game as one's own. It was derived as a two-player, single-board variant of bughouse chess.

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

## Roadmap 

### Search
- [x] Iterative Deepening
- [x] Quiescence Search
- [ ] Aspiration Windows
     
### Move Ordering

- [x] Transposition Table Move
- [x] Killer Move Heuristic
- [ ] Static Exchange Evaluation
    - [x] MVV-LVA

# Evaluation


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
