# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A data structure used to hold / inspect search data for a batch of inputs."""

from __future__ import annotations
from typing import Any, ClassVar, Generic, TypeVar

import chex
import jax
import jax.numpy as jnp


T = TypeVar("T")


@chex.dataclass(frozen=True)
class Tree(Generic[T]):
  """State of a search tree.

  The `Tree` dataclass is used to hold and inspect search data for a batch of
  inputs. In the fields below `B` denotes the batch dimension, `N` represents
  the number of nodes in the tree, and `num_actions` is the number of discrete
  actions.

  node_visits: `[B, N]` the visit counts for each node.
  raw_values: `[B, N]` the raw value for each node.
  node_values: `[B, N]` the cumulative search value for each node.
  parents: `[B, N]` the node index for the parents for each node.
  action_from_parent: `[B, N]` action to take from the parent to reach each
    node.
  children_index: `[B, N, num_actions]` the node index of the children for each
    action.
  children_prior_logits: `[B, N, Anum_actions` the action prior logits of each
    node.
  children_visits: `[B, N, num_actions]` the visit counts for children for
    each action.
  children_rewards: `[B, N, num_actions]` the immediate reward for each action.
  children_discounts: `[B, N, num_actions]` the discount between the
    `children_rewards` and the `children_values`.
  children_values: `[B, N, num_actions]` the value of the next node after the
    action.
  embeddings: `[B, N, ...]` the state embeddings of each node.
  root_invalid_actions: `[B, num_actions]` a mask with invalid actions at the
    root. In the mask, invalid actions have ones, and valid actions have zeros.
  extra_data: `[B, ...]` extra data passed to the search.
  """
  node_visits: chex.Array  # [B, N]
  raw_values: chex.Array  # [B, N]
  node_values: chex.Array  # [B, N]
  parents: chex.Array  # [B, N]
  action_from_parent: chex.Array  # [B, N]
  children_index: chex.Array  # [B, N, num_actions]
  children_prior_logits: chex.Array  # [B, N, num_actions]
  children_visits: chex.Array  # [B, N, num_actions]
  children_rewards: chex.Array  # [B, N, num_actions]
  children_discounts: chex.Array  # [B, N, num_actions]
  children_values: chex.Array  # [B, N, num_actions]
  embeddings: Any  # [B, N, ...]
  root_invalid_actions: chex.Array  # [B, num_actions]
  extra_data: T  # [B, ...]

  # The following attributes are class variables (and should not be set on
  # Tree instances).
  ROOT_INDEX: ClassVar[int] = 0
  NO_PARENT: ClassVar[int] = -1
  UNVISITED: ClassVar[int] = -1

  @property
  def num_actions(self):
    return self.children_index.shape[-1]

  @property
  def num_simulations(self):
    return self.node_visits.shape[-1] - 1

  def qvalues(self, indices):
    """Compute q-values for any node indices in the tree."""
    # pytype: disable=wrong-arg-types  # jnp-type
    if jnp.asarray(indices).shape:
      return jax.vmap(_unbatched_qvalues)(self, indices)
    else:
      return _unbatched_qvalues(self, indices)
    # pytype: enable=wrong-arg-types

def infer_batch_size(tree: Tree) -> int:
  """Recovers batch size from `Tree` data structure."""
  if tree.node_values.ndim != 2:
    raise ValueError("Input tree is not batched.")
  chex.assert_equal_shape_prefix(jax.tree_util.tree_leaves(tree), 1)
  return tree.node_values.shape[0]

def _unbatched_qvalues(tree: Tree, index: int) -> int:
  chex.assert_rank(tree.children_discounts, 2)
  return (  # pytype: disable=bad-return-type  # numpy-scalars
      tree.children_rewards[index]
      + tree.children_discounts[index] * tree.children_values[index]
  )
