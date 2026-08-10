from pathlib import Path

from scripts.train_from_games_parquet import _run_supervised_training


def test_supervised_launcher_forwards_staged_dataset_and_model_options(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_run(command, cwd, check):
        captured['command'] = command
        captured['cwd'] = cwd
        captured['check'] = check

    monkeypatch.setattr(
        'scripts.train_from_games_parquet.subprocess.run',
        fake_run,
    )
    train_planes = tmp_path / 'planes' / 'train'
    validation = tmp_path / 'planes' / 'val' / 'evaluation_shard.parquet'
    train_evaluation = (
        tmp_path / 'planes' / 'train_eval' / 'evaluation_shard.parquet'
    )

    _run_supervised_training(
        tmp_path,
        train_planes_dir=train_planes,
        validation_shard=validation,
        architecture='crossboard-risev33',
        batch_size=256,
        precision='bf16',
        train_eval_shard=train_evaluation,
    )

    command = captured['command']
    assert command[1:4] == ['train_loop.py', '--mode', 'sl']
    assert command[command.index('--architecture') + 1] == 'crossboard-risev33'
    assert command[command.index('--batch-size') + 1] == '256'
    assert command[command.index('--precision') + 1] == 'bf16'
    assert command[command.index('--sl-train-data-dir') + 1] == str(train_planes)
    assert command[command.index('--sl-validation-shard') + 1] == str(validation)
    assert command[command.index('--train-eval-shard') + 1] == str(train_evaluation)
    assert captured['cwd'] == tmp_path / 'src' / 'training'
    assert captured['check'] is True