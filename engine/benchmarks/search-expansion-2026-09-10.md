# Search expansion investigation: 2026-09-10

## Retained changes

- `Node::expand_next_joint_child` now consumes its candidate only after
  successfully reserving the child's evaluation. Passing a busy transposition
  previously discarded the action and desynchronized generated-action and
  child-edge indices. The new regression fails before the fix and passes after
  it. The current production caller passes a fresh child, then canonicalizes
  transpositions separately, so this is an API safeguard, not an explanation
  for the observed game loss or a demonstrated live strength improvement.
- Node initialization borrows the input policy vectors unless root noise is
  enabled. This removes two vector allocations and copies per ordinary leaf.
  Root noise still operates on private copies. Priors, legality, exploration
  parameters, and mate-probe restrictions are unchanged.

## Rejected experiment

Game 183311301489 suggested insufficient exploration of attacks that wait on
one board. A bounded experiment promoted one legal single-board wait per
board, leaving its prior unchanged. It passed candidate-generator tests but
did not improve recognition at a 60,000-node allowance in three replayed
positions: before Qxc4, before B@b1, and the opponent's attack after B@b1.
The heuristic and its experimental test were removed.

Direct policy inspection after B@b1 showed that waiting was not nearly absent:
board A pass had probability 0.0980 and board B pass 0.3764. The joint action
P@h7/pass had a substantial factorized prior. The game does not establish a
need for a general pass-prior floor.

## Throughput check

Compared the pre-investigation executable, including the earlier mate fixes,
with the retained changes. Used the TensorRT FP16 iteration-04 model, batch
size 16, default workers, no opening noise, draw contempt 1000, and raw UCI
`QEarlyExit=false`. Each search started after `ucinewgame` and requested
120,000 nodes. Three samples per binary and position used alternating order.

| Position                     | Baseline median NPS | Changed median NPS |
| ---------------------------- | ------------------: | -----------------: |
| Opening after board A e2e4   |              28,059 |             26,143 |
| Game 183311301489, seq 16611 |              27,050 |             26,078 |

Samples ranged from roughly 23,000 to 28,000 NPS across both executables.
These measurements do not demonstrate a throughput gain; the changed-binary
medians were lower. Best moves remained d7d5/e2e4 and pass/a3b4 respectively.
Removing allocations is a local cost reduction, not evidence of better
end-to-end speed or playing strength. No Elo gain or fix for the game's missed
attack is claimed.

## Focused verification

```sh
cmake --build engine/build-ninja --target hivemind_tests hivemind -j 8
engine/build-ninja/tests/hivemind_tests \
  --gtest_filter='NodeTest.*:JointActionTest.*:JointCandidateGeneratorTest.*:*Noise*'
```

The full-suite baseline has two known failures:
`PolicyTest.NormalizesExtremeAndNonFiniteLogits` and
`BackendCompatTest.HalfPreservesInfinityAndNan`.
