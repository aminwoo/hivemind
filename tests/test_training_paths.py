from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from configs.main_config import main_config
import src.training.trainer_agent as trainer_agent_module
from src.training.train_loop import (
    _resolve_train_evaluation_shard,
    _resolve_validation_shard,
    _checkpoint_progress,
    configure_batch_size,
    configure_precision,
    configure_supervised_evaluation,
    project_root,
)
from configs.train_config import TrainConfig
from src.training.trainer_agent import (
    TrainerAgentPytorch,
    evaluation_batch_limit,
    requires_full_evaluation,
    save_torch_state,
)


def test_default_plane_paths_are_anchored_to_project_root():
    assert Path(main_config['planes_train_dir']) == project_root / 'data' / 'planes' / 'train'
    assert Path(main_config['planes_val_dir']) == project_root / 'data' / 'planes' / 'val'


def test_default_validation_shard_exists():
    assert Path(_resolve_validation_shard()).is_file()


def test_explicit_validation_shard_is_used(tmp_path):
    validation_shard = tmp_path / 'validation.parquet'
    validation_shard.touch()
    assert Path(_resolve_validation_shard(str(validation_shard))) == validation_shard


def test_train_evaluation_falls_back_to_first_training_shard():
    resolved = Path(_resolve_train_evaluation_shard())
    expected = sorted((project_root / 'data' / 'planes' / 'train').glob('*.parquet'))[0]
    assert resolved == expected


def test_explicit_missing_train_evaluation_shard_fails(tmp_path):
    missing = tmp_path / 'missing.parquet'
    with pytest.raises(FileNotFoundError, match=str(missing)):
        _resolve_train_evaluation_shard(str(missing))


def test_cross_board_batch_size_preserves_evaluation_sample_cadence():
    train_config = TrainConfig()
    original_evaluation_samples = (
        train_config.batch_size * train_config.batch_steps
    )

    configure_batch_size(train_config, 'crossboard-risev33')

    assert train_config.batch_size == 256
    assert (
        train_config.batch_size * train_config.batch_steps
        == original_evaluation_samples
    )


def test_dual_stream_batch_size_preserves_evaluation_sample_cadence():
    train_config = TrainConfig()
    original_evaluation_samples = (
        train_config.batch_size * train_config.batch_steps
    )

    configure_batch_size(train_config, 'dualstream-memory-risev33')

    assert train_config.batch_size == 256
    assert (
        train_config.batch_size * train_config.batch_steps
        == original_evaluation_samples
    )


def test_dual_stream_defaults_to_bf16_with_fp32_override():
    train_config = TrainConfig()

    configure_precision(train_config, 'dualstream-memory-risev33')
    assert train_config.mixed_precision == 'bf16'

    configure_precision(train_config, 'dualstream-memory-risev33', 'fp32')
    assert train_config.mixed_precision == 'fp32'


def test_rl_defaults_to_bf16_for_crossboard_with_fp32_override():
    train_config = TrainConfig()

    configure_precision(train_config, 'crossboard-risev33', is_rl=True)
    assert train_config.mixed_precision == 'bf16'

    configure_precision(
        train_config,
        'crossboard-risev33',
        'fp32',
        is_rl=True,
    )
    assert train_config.mixed_precision == 'fp32'


def test_supervised_evaluation_is_lighter_and_less_frequent():
    train_config = TrainConfig()
    original_evaluation_samples = (
        train_config.batch_size * train_config.batch_steps
    )

    configure_supervised_evaluation(train_config)
    configure_batch_size(train_config, 'crossboard-risev33')

    assert train_config.eval_batches == 64
    assert train_config.full_eval_each_epoch is True
    assert (
        train_config.batch_size * train_config.batch_steps
        == original_evaluation_samples * 2
    )


def test_full_evaluation_removes_intermediate_batch_cap():
    train_config = TrainConfig(eval_batches=64)

    assert evaluation_batch_limit(train_config, full_evaluation=False) == 64
    assert evaluation_batch_limit(train_config, full_evaluation=True) is None


def test_epoch_boundary_requires_full_supervised_evaluation():
    train_config = TrainConfig(full_eval_each_epoch=True)

    assert requires_full_evaluation(train_config, False, True) is True
    assert requires_full_evaluation(train_config, False, False) is False
    assert requires_full_evaluation(TrainConfig(), False, True) is False
    assert requires_full_evaluation(TrainConfig(), True, False) is True


@pytest.mark.parametrize(
    ("full_evaluation", "expected_batch_limit"),
    [(False, 64), (True, None)],
)
def test_trainer_passes_expected_batch_limit_to_evaluators(
    monkeypatch,
    full_evaluation,
    expected_batch_limit,
):
    observed_batch_limits = []

    def fake_evaluate_metrics(*args, nb_batches, **kwargs):
        observed_batch_limits.append(nb_batches)
        return {
            "loss": torch.tensor(1.0),
            "policy_acc": torch.tensor(0.0),
        }

    class Model:
        def train(self):
            return self

    monkeypatch.setattr(
        trainer_agent_module,
        "evaluate_metrics",
        fake_evaluate_metrics,
    )
    trainer = object.__new__(TrainerAgentPytorch)
    trainer.tc = TrainConfig(
        batch_steps=2000,
        eval_batches=64,
        context="cpu",
    )
    trainer.to = SimpleNamespace(
        metrics={},
        phase_weights={},
        lr_schedule=lambda _: 0.01,
        momentum_schedule=lambda _: 0.9,
    )
    trainer.batch_proc_tmp = trainer.tc.batch_steps
    trainer.t_s_steps = 0
    trainer.k_steps = 0
    trainer.k_steps_end = 1
    trainer.patience_cnt = 0
    trainer.cur_it = 1
    trainer._train_eval_loader = object()
    trainer._val_loader = object()
    trainer._model = Model()
    trainer._ctx = torch.device("cpu")
    trainer.additional_loaders = None

    trainer.evaluate(object(), full_evaluation=full_evaluation)

    assert observed_batch_limits == [
        expected_batch_limit,
        expected_batch_limit,
    ]


def test_explicit_batch_size_overrides_architecture_default():
    train_config = TrainConfig()
    configure_batch_size(train_config, 'crossboard-risev33', 128)
    assert train_config.batch_size == 128


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        configure_batch_size(TrainConfig(), 'crossboard-risev33', 0)


def test_checkpoint_progress_uses_explicit_metadata():
    assert _checkpoint_progress({
        "training_iteration": 93_417,
        "evaluation_step": 24,
    }) == (93_417, 24)


def test_checkpoint_progress_rejects_missing_metadata():
    with pytest.raises(
        ValueError,
        match="missing required progress metadata: evaluation_step",
    ):
        _checkpoint_progress({"training_iteration": 93_417})


def test_new_checkpoint_records_exact_training_progress(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "checkpoint.tar"

    save_torch_state(
        model,
        optimizer,
        checkpoint_path,
        training_iteration=93_417,
        evaluation_step=24,
        batch_steps=8_000,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["training_iteration"] == 93_417
    assert checkpoint["evaluation_step"] == 24
    assert checkpoint["batch_steps"] == 8_000