import struct

import numpy as np
import polars as pl
import pytest
import torch

from src.preprocessing.convert_selfplay_data import (
    DEFAULT_RL_VALIDATION_FRACTION,
    HEADER,
    NB_INPUT_VALUES,
    POLICY_ENTRY,
    SAMPLE_METADATA,
    convert_to_split_parquet,
    is_validation_game,
)
from src.training.trainer_agent import TrainerAgentPytorch
from src.training.train_loop import (
    RL_WEIGHTS_DIR,
    SUPERVISED_WEIGHTS_DIR,
    save_final_rl_checkpoint,
)


def _write_chunk(path, game_ids):
    with path.open("wb") as output:
        output.write(HEADER.pack(b"HVM3", 3, 74, 4672, len(game_ids)))
        for game_id in game_ids:
            output.write(SAMPLE_METADATA.pack(game_id, 100, 1, 20, 0, 0, 1, 2))
            output.write(bytes(NB_INPUT_VALUES))
            for _ in range(2):
                output.write(struct.pack("<H", 1))
                output.write(POLICY_ENTRY.pack(0, np.float32(1.0)))


def test_split_conversion_is_deterministic_and_game_disjoint(tmp_path):
    input_dir = tmp_path / "training_data"
    input_dir.mkdir()

    validation_game = next(
        game_id
        for game_id in range(1000)
        if is_validation_game("run", game_id, 0.5, 7)
    )
    training_game = next(
        game_id
        for game_id in range(1000)
        if not is_validation_game("run", game_id, 0.5, 7)
    )
    _write_chunk(input_dir / "chunk_run_000000.hvm", [validation_game, training_game])
    _write_chunk(input_dir / "chunk_run_000001.hvm", [validation_game, training_game])

    output_dir = tmp_path / "rl_data"
    train_dir, val_dir, train_count, val_count = convert_to_split_parquet(
        input_dir,
        output_dir,
        validation_fraction=0.5,
        split_seed=7,
        samples_per_shard=1,
    )

    train_rows = pl.read_parquet(train_dir / "*.parquet")
    val_rows = pl.read_parquet(val_dir / "*.parquet")
    assert train_count == len(train_rows) == 2
    assert val_count == len(val_rows) == 2
    assert set(train_rows["game_id"]) == {training_game}
    assert set(val_rows["game_id"]) == {validation_game}
    assert len(list(input_dir.glob("*.hvm"))) == 2

    stale_file = output_dir / "train" / "stale.parquet"
    stale_file.touch()
    convert_to_split_parquet(input_dir, output_dir, 0.5, 7, samples_per_shard=4)
    assert not stale_file.exists()


def test_checkpoint_cleanup_preserves_unrelated_weights(tmp_path):
    seed_checkpoint = tmp_path / "seed.tar"
    generated_checkpoint = tmp_path / "generated.tar"
    seed_checkpoint.touch()
    generated_checkpoint.touch()

    trainer = TrainerAgentPytorch.__new__(TrainerAgentPytorch)
    trainer._generated_weight_files = [generated_checkpoint]
    trainer.delete_previous_weights()

    assert seed_checkpoint.exists()
    assert not generated_checkpoint.exists()


def test_split_conversion_rejects_output_containing_source(tmp_path):
    input_dir = tmp_path / "training_data"
    input_dir.mkdir()
    source_chunk = input_dir / "chunk_run_000000.hvm"
    _write_chunk(source_chunk, [1])

    with pytest.raises(ValueError, match="must not contain"):
        convert_to_split_parquet(input_dir, tmp_path)

    assert source_chunk.exists()


def test_default_validation_fraction_is_two_percent():
    assert DEFAULT_RL_VALIDATION_FRACTION == 0.02


def test_model_artifacts_use_separate_directories():
    assert RL_WEIGHTS_DIR.name == "rl"
    assert SUPERVISED_WEIGHTS_DIR.name == "supervised"
    assert RL_WEIGHTS_DIR.parent == SUPERVISED_WEIGHTS_DIR.parent


def test_final_rl_checkpoint_is_resumable(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    checkpoint_path = save_final_rl_checkpoint(model, optimizer, tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    assert checkpoint_path.name == "model-rl-final.tar"
    assert set(checkpoint) == {"model_state_dict", "optimizer_state_dict"}