import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest

from test_distill import _write_chunk
from test_rl_pilot import executable


ROOT = Path(__file__).resolve().parents[2]
GAME = 'rlfakegame'  # a process name (at most 15 characters)


@pytest.fixture
def loop(tmp_path):
    """The loop script in tmp_path, with fake engine and trainer binaries."""
    script = tmp_path / 'loop.sh'
    script.write_text((ROOT / 'data/distill/rl_loop.sh').read_text().replace(
        'cd /home/ben/hivemind', f'cd {shlex.quote(str(tmp_path))}'))
    for name, text in (('artifacts/distill/twin-s-noattn/best.pt', 'it1-pt'),
                       ('engine/models/twin-s-noattn-sp1.onnx', 'it1'),
                       ('engine/models/twin-s-noattn.onnx', 'it1'),
                       ('engine/models/twin-s-noattn-it0.onnx', 'it0'),
                       ('data/distill/openings.tsv', 'openings'),
                       ('data/distill/runs/rl-it1/arms/C/last.pt', 'it1-pt')):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (tmp_path / 'data/distill/val').mkdir()
    (tmp_path / 'data/distill/runs/rl-it1/search/train').mkdir(parents=True)
    _write_chunk(tmp_path / 'fixture.dst', np.random.default_rng(31))
    _write_chunk(tmp_path / 'data/distill/runs/rl-it1/search/train/search_31_00000.dst',
                 np.random.default_rng(31))
    executable(tmp_path / '.venv/bin/python', f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    # The trainer records its arguments; its "model" names the run it came from.
    executable(tmp_path / '.venv/bin/hivemind', f'''#!{sys.executable}
import json, sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index('--out') + 1])
out.mkdir()
(out / 'args.json').write_text(json.dumps(sys.argv[1:]))
(out / 'last.pt').write_text('pt ' + str(out))
(out / 'distill-twin-s-v3.0.onnx').write_text('model ' + str(out))
''')
    # Self-play copies the fixture chunk; FAIL_SEED fails that seed. A contender
    # from FAKE_LOSER loses every game, others win. FAKE_GAME_ONCE launches a
    # game the first time a match starts, so the loop must restart it.
    executable(tmp_path / 'engine/build-sp1/hivemind', f'''#!{sys.executable}
import os, shutil, sys
from pathlib import Path
args = sys.argv
def option(name): return args[args.index(name) + 1]
with open('../calls.log', 'a') as log: log.write('selfplay ' + option('--seed') + '\\n')
if option('--seed') == os.environ.get('FAIL_SEED'): sys.exit(1)
Path(option('--output')).mkdir(parents=True)
data = Path(option('--distill-output'))
data.mkdir(parents=True)
shutil.copy('../fixture.dst', data / 'search.dst')
n = int(option('--games'))
for i in range(n): print(f'selfplay game {{i+1}}/{{n}} termination checkmate')
''')
    executable(tmp_path / 'engine/build-ninja/hivemind', f'''#!{sys.executable}
import json, os, subprocess, sys, time
from pathlib import Path
args = sys.argv
def option(name): return args[args.index(name) + 1]
with open('../calls.log', 'a') as log: log.write('match ' + option('--baseline') + '\\n')
flag = Path('../game-launched')
if os.environ.get('FAKE_GAME_ONCE') and not flag.exists():
    flag.write_text('')
    game = Path('../{GAME}')  # a script's process is named after its file
    game.write_text('#!/bin/sh\\nsleep 3\\n')
    game.chmod(0o755)
    subprocess.Popen([str(game)], start_new_session=True)
    time.sleep(30)
out = Path(option('--output'))
out.mkdir(parents=True)
games = int(option('--games'))
lose = os.environ.get('FAKE_LOSER', '\\0') in option('--contender')
d = dict(contender_wins=0 if lose else games, baseline_wins=games if lose else 0, draws=0,
         contender_elo=None, elo_confidence_95=None,
         performance=dict(contender=dict(nps=1), baseline=dict(nps=1)))
(out / 'summary.json').write_text(json.dumps(d))
''')

    def run(*extra, **env):
        environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), RL_GAMING_PATTERN=GAME,
                           RL_QUIET_SECONDS='1', RL_POLL_SECONDS='1', RL_RETRY_SECONDS='0', **env)
        return subprocess.run(['bash', str(script), '--games', '5', '--segment-games', '2',
                               '--val-games', '2', '--match-games', '4', *extra],
                              env=environment, capture_output=True, text=True, timeout=120)
    return tmp_path, run


def stages(path, n):
    return {p.stem: p.read_text() for p in (path / f'data/distill/runs/rl-it{n}/stages').glob('*.done')}


def test_iterations_replay_older_data_and_promote_only_winners(loop):
    path, run = loop
    runs = path / 'data/distill/runs'
    result = run('--last', '3', FAKE_LOSER='rl-it3')
    assert result.returncode == 0, result.stdout + result.stderr

    it2 = json.loads((runs / 'rl-it2/arms/C/args.json').read_text())
    def values(args, flag):
        start = args.index(flag) + 1
        end = next((i for i in range(start, len(args)) if args[i].startswith('--')), len(args))
        return args[start:end]
    assert values(it2, '--init') == [str(runs / 'rl-it1/arms/C/last.pt')]
    assert values(it2, '--search-data') == [str(runs / f'rl-it2/search/train/seg0{k}') for k in range(3)]
    assert values(it2, '--replay-data') == [str(runs / 'rl-it1/search/train')]
    # 3 new chunks of 16 positions x 1.5 and one old chunk x 0.5: 80 samples, 10% replay.
    assert values(it2, '--replay-fraction') == ['0.1'] and values(it2, '--steps') == ['1']
    assert values(it2, '--search-val') == [str(runs / 'rl-it2/search/val')]
    it3 = json.loads((runs / 'rl-it3/arms/C/args.json').read_text())
    assert values(it3, '--init') == [str(runs / 'rl-it2/arms/C/last.pt')]
    assert values(it3, '--replay-data') == ([str(runs / f'rl-it2/search/train/seg0{k}') for k in range(3)]
                                            + [str(runs / 'rl-it1/search/train')])

    # it2 won and became the generator; it3 lost, so it2 still generates.
    assert json.loads(stages(path, 2)['promote'])['promoted'] is True
    assert json.loads(stages(path, 3)['promote'])['promoted'] is False
    winner = (runs / 'rl-it2/models/C.onnx').read_text()
    for name in ('twin-s-noattn.onnx', 'twin-s-noattn-sp1.onnx'):
        assert (path / 'engine/models' / name).read_text() == winner
    assert (path / 'artifacts/distill/twin-s-noattn/best.pt').read_text() == 'pt ' + str(runs / 'rl-it2/arms/C')
    assert (path / 'engine/models/twin-s-noattn-it3.onnx').read_text() == (runs / 'rl-it3/models/C.onnx').read_text()
    # Each iteration's self-play used the generator at its start.
    assert (runs / 'rl-it3/stages/generator').read_text() != (runs / 'rl-it2/stages/generator').read_text()
    for n in (2, 3):
        assert {'seg00', 'seg01', 'seg02', 'val', 'train', 'match_generator', 'match_it0',
                'promote', 'iteration'} <= set(stages(path, n))
        assert json.loads(stages(path, n)['match_it0'])['score'] == (0.0 if n == 3 else 1.0)

    # Finished iterations are skipped; the next run starts at it4.
    calls = (path / 'calls.log').read_text()
    assert run('--last', '3').returncode == 0
    assert (path / 'calls.log').read_text() == calls


def test_failed_segment_is_retried_then_resumed_without_redoing_finished_ones(loop):
    path, run = loop
    result = run('--last', '2', FAIL_SEED='2001')
    assert result.returncode != 0
    assert 'attempt 3 of 3' in result.stdout
    calls = (path / 'calls.log').read_text().split('\n')
    assert calls.count('selfplay 2000') == 1 and calls.count('selfplay 2001') == 3
    assert not (path / 'data/distill/runs/rl-it2/stages/train.done').exists()

    result = run('--last', '2')
    assert result.returncode == 0, result.stdout + result.stderr
    calls = (path / 'calls.log').read_text().split('\n')
    assert calls.count('selfplay 2000') == 1 and calls.count('selfplay 2001') == 4
    assert {'seg00', 'seg01', 'seg02', 'iteration'} <= set(stages(path, 2))


def test_stop_file_exits_before_the_next_stage(loop):
    path, run = loop
    (path / 'data/distill/runs').mkdir(parents=True, exist_ok=True)
    (path / 'data/distill/runs/rl-loop.stop').write_text('')
    result = run('--last', '2')
    assert result.returncode == 0 and 'stop requested' in result.stdout
    assert not (path / 'calls.log').exists()


def test_match_restarts_when_a_game_starts(loop):
    path, run = loop
    result = run('--last', '2', FAKE_GAME_ONCE='1')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'a game started, restarting the match' in result.stdout
    assert 'waiting: a game is running' in result.stdout
    calls = (path / 'calls.log').read_text().split('\n')
    assert sum(c.startswith('match') for c in calls) == 3  # generator twice, it0 once
    assert list((path / 'data/distill/runs/rl-it2/aborted').glob('arms_match_generator.log-*'))


def test_rejects_directories_it_did_not_make(loop):
    path, run = loop
    (path / 'data/distill/runs/rl-it2').mkdir(parents=True)
    result = run('--last', '2')
    assert result.returncode != 0 and 'not made by this loop' in result.stdout
