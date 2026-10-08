# Bughouse search tuning

The 2026-10-06 campaign compares probability-mass widening and nearby search
settings using `twin-s-noattn.onnx`. Testing was stopped on 2026-10-07 at the
user's request, and the best completed screening configuration was adopted:
internal count-widening coefficient 1, root coefficient 4, exponent 0.3,
with mass widening disabled.

## Adopted result

Narrower count widening scored 1,020 wins, 970 losses, and 10 draws across
2,000 games at 30 ms/move: 51.25%, or +8.69 Elo with a paired 95% interval
of +4.28 to +13.10. This changes only the internal coefficient from 2 to 1.
The root coefficient and all other search defaults retain their previous
values. Independent validation and the held-out match were not run.

Nine screening comparisons completed. Wider count widening was interrupted
after 1,083 games and is retained as partial data. All results and the original
coefficient-2 engine snapshot remain in the campaign directory.

## Widening experiment

The baseline allows `ceil(c * N^e)` children: `c=2` internally, `c=4` at the
root, and `e=0.3`. Mass widening instead expands until the generated children
cover `1 - (1 - start) * (N + 1)^(-massExponent)` of the prior distribution,
subject to `ceil(massCap * countLimit)` children. It still visits generated
children before adding another and widens immediately when every expanded
child is proven losing.

The original implementation measures absolute joint prior mass. Joint actions
that violate sitting rules can consume part of that distribution, leaving
less than one total legal mass. The experimental `pw-mass-normalize` setting
measures coverage relative to the total legal joint mass. That total is summed
from each board's sit, quiet-move, and capture probabilities in O(A+B) time;
learned residual rescoring preserves the total. Zero legal probability falls
back to count widening.

`root-pw-mass` sets an independent root target. `-1` inherits the internal
target; `0` uses count widening at the root. Both features are opt-in.

The previous `.35/.175/2` mass schedule scored 153 wins, 162 losses, and five
draws over 320 games at 100 ms/move: about −10 Elo, with a 95% interval of
−48 to +28. That result did not establish a gain or loss.

## Campaign protocol

1. Generate teacher-policy games with seed `2026100602`, temperature sampling,
   3% random moves, and a 40-macro-ply limit. These games supply openings only;
   the match games retain the normal 400-macro-ply limit.
2. Sample one opening between macro plies 6 and 18 from each source game,
   deduplicate, and shuffle with seed `20261006`. The resulting 5,165 openings
   are split into 1,000 discovery, 1,000 validation, and 3,165 held-out openings.
   Each comparison uses 1,000 different openings, played in color-swapped pairs.
3. Screen 16 configurations for 2,000 games each at 30 ms/move. Configurations
   include an A/A control, legacy and normalized mass widening, independent
   root targets, narrower/wider count schedules, CPUCT 2 and 4, disabling the
   moves-left discount, 25% WDL blending with WDL enabled, and Q weight 0.5.
4. Validate the top three non-control candidates for 2,000 games each at
   60 ms/move on unused openings. Select one finalist using validation score.
5. Test that one finalist for 2,000 games at 100 ms/move on held-out openings.
   Discovery and validation estimates are selection data. Only the final
   independent comparison is used to judge a gain. The campaign reports
   support for a gain only if its paired 95% score interval is above 50%.

The full expanded campaign planned 40,000 games; it was stopped before
screening finished. The original 128-game comparisons are
preserved in `engine/tournament_results/search-tuning-20261006/campaign` as
preliminary data; they are not combined with the expanded campaign's results.

Each opening is played twice with contestant colors swapped. Both sides use
four search workers and inference batch 8. Per-move Dirichlet noise is off.
Both contestants use the same immutable experimental engine binary and the
same model. Their settings differ only in the explicitly recorded options.

The controller records engine, model, script, and opening SHA-256 hashes, every
command, per-game PGNs, results, effective widening/CPUCT settings, and search
throughput. Completed-run markers prevent a partial match from being treated
as a finished result during resumption. Explicit root coefficient overrides
are applied after the legacy internal coefficient flag, which also sets the
root coefficient.
The expanded job uses frozen copies of both Python scripts and checks the
model hash before each comparison.

## Run and inspect

From the repository root, first build an isolated Release TensorRT executable
(the updated default is built in `engine/build-tuning/hivemind.bin`). Keep the
BOT's executable separate. To resume the stopped historical campaign, use its
frozen controller and coefficient-2 binary as shown below; a new build uses
the adopted coefficient-1 default.

```sh
engine/build-tuning/hivemind.bin gennnue \
  --model engine/models/twin-s-noattn.onnx --positions 200000 \
  --threads 1 --batch-size 8 --seed 2026100602 --max-macro-plies 40 \
  --random-move-prob 0.03 --fens true \
  --output engine/tournament_results/search-tuning-20261006/opening-source-2000

python3 engine/tournament_results/search-tuning-20261006/campaign-2000/controller.py \
  --engine engine/tournament_results/search-tuning-20261006/campaign-2000/hivemind-tuning.bin \
  --model engine/models/twin-s-noattn.onnx \
  --opening-source engine/tournament_results/search-tuning-20261006/opening-source-2000 \
  --screen-games 2000 --validate-games 2000 --holdout-games 2000 \
  --output engine/tournament_results/search-tuning-20261006/campaign-2000
```

Re-running the controller with identical arguments resumes completed matches.
An interrupted match starts again; it is not combined with partial results.
Change output directories when changing campaign inputs or budgets.

The stopped job used the user service
`hivemind-bughouse-tuning-2000-20261006.service`. Its saved results can be read with:

```sh
tail /tmp/hivemind-tuning-campaign-2000.log
cat engine/tournament_results/search-tuning-20261006/campaign-2000/report.md
```

During a run, the report updates after each completed comparison and
`campaign.json` records the active comparison. The stopped campaign is marked
as stopped; it will not resume automatically.

To reproduce a configuration in UCI, use `PWMassStartPermille`,
`RootPWMassStartPermille`, `PWMassExponentPermille`, `PWMassCapPermille`,
`PWMassNormalize`, and `CPUCTInitPermille`, alongside the existing count
widening options. `RootPWMassStartPermille=-1` means inherit, not −0.001.

## Interpretation limits

The final interval uses the tournament's paired-opening normal approximation.
Fixed-time four-worker search is nondeterministic; hardware load can affect
throughput. The A/A control checks for a large side bias. Matches reset search
state between moves and keep each team's time advantage fixed, as the existing
Bughouse tournament does. Results therefore apply to this selfplay protocol
and budget; a positive result should also be checked with retained-tree play
and longer move times before treating it as a universal improvement.
