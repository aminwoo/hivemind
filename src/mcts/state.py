from typing import Optional
import chex
from chex import dataclass
import jax.numpy as jnp
import jax
from src.mcts.evaluator import EvalOutput
from src.mcts.tree import Tree

@dataclass(frozen=True)
class MCTSNode:
    n: jnp.number
    p: chex.Array
    q: jnp.number
    terminated: jnp.number
    embedding: chex.ArrayTree

    @property
    def w(self) -> jnp.number:
        return self.q * self.n


MCTSTree = Tree[MCTSNode] 

@dataclass(frozen=True)
class TraversalState:
    parent: int
    action: int

@dataclass(frozen=True)
class BackpropState:
    node_idx: int
    value: float
    tree: MCTSTree

@dataclass(frozen=True)
class MCTSOutput(EvalOutput):
    eval_state: MCTSTree
    policy_weights: chex.Array
