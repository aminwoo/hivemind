import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest

from test_distill import _write_chunk


ROOT = Path(__file__).resolve().parents[2]


def executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


@pytest.fixture
def pilot(tmp_path):
    script = tmp_path / 'pilot.sh'
    script.write_text((ROOT / 'data/distill/rl_pilot.sh').read_text().replace(
        'cd /home/ben/hivemind', f'cd {shlex.quote(str(tmp_path))}'))
    for name in ('artifacts/distill/twin-s-noattn/best.pt',
                 'engine/models/twin-s-noattn-sp1.onnx', 'engine/models/twin-s-noattn.onnx',
                 'openings.tsv'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    saved_script = tmp_path / 'data/distill/rl_pilot.sh'
    saved_script.parent.mkdir(parents=True, exist_ok=True)
    saved_script.write_text(script.read_text())
    _write_chunk(tmp_path / 'fixture.dst', np.random.default_rng(31))
    for name in ('train', 'val'):
        (tmp_path / 'data/distill' / name).mkdir()
    executable(tmp_path / '.venv/bin/python', f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    executable(tmp_path / '.venv/bin/hivemind', '''#!/usr/bin/env python3
import os, sys
from pathlib import Path
if os.environ.get('FAIL') == 'train': sys.exit(1)
out = Path(sys.argv[sys.argv.index('--out') + 1])
out.mkdir()
(out / 'last.pt').write_text('checkpoint')
(out / 'distill-twin-s-v3.0.onnx').write_text('model')
''')
    engine = '''#!/usr/bin/env python3
import json, os, shutil, sys
from pathlib import Path
args = sys.argv
def option(name): return args[args.index(name) + 1]
out = Path(option('--output'))
out.mkdir(parents=True)
if args[1] == 'selfplay':
    if os.environ.get('FAIL') == 'selfplay': sys.exit(1)
    data = Path(option('--distill-output'))
    data.mkdir()
    shutil.copy('../fixture.dst', data / 'search_31_00000.dst')
    n = int(option('--games')) - (os.environ.get('FAIL') == 'game-count')
    for i in range(n): print(f'selfplay game {i+1}/{n} termination draw')
else:
    if os.environ.get('FAIL') == 'tournament': sys.exit(1)
    n = 159 if os.environ.get('FAIL') == 'summary-count' else 160
    d = dict(contender_wins=0, baseline_wins=0, draws=n, contender_elo=0,
             elo_confidence_95=[-10, 10], performance=dict(contender=dict(nps=1), baseline=dict(nps=1)))
    if os.environ.get('FAIL') != 'summary-missing':
        (out / 'summary.json').write_text(json.dumps(d))
'''
    for name in ('build-sp1', 'build-ninja'):
        executable(tmp_path / 'engine' / name / 'hivemind', engine)
    executable(tmp_path / 'bin/cp', '''#!/bin/sh
[ "${FAIL:-}" != copy ] || exit 1
exec /bin/cp "$@"
''')

    def run(failure='', arm='A'):
        env = dict(os.environ, FAIL=failure, PYTHONPATH=str(ROOT / 'src'),
                   PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}")
        return subprocess.run(['bash', str(script), '--games', '3', '--val-games', '2',
                               '--arm', arm, '--out', str(tmp_path / 'run'),
                               '--openings', str(tmp_path / 'openings.tsv')],
                              env=env, capture_output=True, text=True)
    return tmp_path, run


@pytest.mark.parametrize('arm', ['A', 'B', 'both'])
def test_pilot_runs_requested_arms_and_rejects_reruns(pilot, arm):
    path, run = pilot
    result = run(arm=arm)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((path / 'run/manifest.json').read_text())
    assert manifest['stage'] == 'finished'
    assert manifest['games'] == 3 and manifest['val_games'] == 2
    assert manifest['files'] and manifest['arms'] == arm
    for name in ('A', 'B'):
        assert (path / f'run/arms/match_{name}/summary.json').exists() == (arm in (name, 'both'))
    assert run(arm=arm).returncode != 0


@pytest.mark.parametrize('failure', ['selfplay', 'game-count', 'train', 'copy',
                                     'tournament', 'summary-count', 'summary-missing'])
def test_pilot_failures_cannot_report_finished(pilot, failure):
    _, run = pilot
    result = run(failure)
    assert result.returncode != 0
    assert 'pilot finished' not in result.stdout
