<div align="center">
  
  ![hivemind-logo](https://github.com/aminwoo/hivemind/assets/124148472/d42c6a6e-ab2e-4d7a-bf90-4876d59c9558)
  
  <h3>HiveMind</h3>

  A free UCI Bughouse engine.

</div>

## Overview
Bughouse is a four player chess variant where players exchange captured pieces which can be dropped on empty squares. The large game tree and complexities around coordination between players makes Bughouse a challenging domain for traditional Chess algorithms. HiveMind is a two board engine which uses a policy network and Monte Carlo tree search (MCTS) to narrow down the search space and coordinate both boards. 

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
