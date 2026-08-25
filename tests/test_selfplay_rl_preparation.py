import struct

import numpy as np
import polars as pl
import pytest
import torch

from src.preprocessing.convert_selfplay_data import (
    DEFAULT_RL_SAMPLES_PER_SHARD,
    DEFAULT_RL_VALIDATION_FRACTION,
    HEADER,
    NB_INPUT_VALUES,
    POLICY_ENTRY,
    JOINT_POLICY_ENTRY,
    MAX_JOINT_POLICY_ENTRIES,
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


def _write_chunk(path, game_ids, version=4):
    magic = {3: b"HVM3", 4: b"HVM4", 5: b"HVM5"}[version]
    with path.open("wb") as output:
        output.write(HEADER.pack(magic, version, 74, 4672, len(game_ids)))
        for game_id in game_ids:
            if version >= 4:
                output.write(SAMPLE_METADATA.pack(game_id, 100, 1, 20, 0, 0, 1, 2, 0.5))
            else:
                from src.preprocessing.convert_selfplay_data import SAMPLE_METADATA_V3
                output.write(SAMPLE_METADATA_V3.pack(game_id, 100, 1, 20, 0, 0, 1, 2))
            output.write(bytes(NB_INPUT_VALUES))
            for _ in range(2):
                output.write(struct.pack("<H", 1))
                output.write(POLICY_ENTRY.pack(0, np.float32(1.0)))
            if version >= 5:
                output.write(struct.pack("<H", 2))
                output.write(JOINT_POLICY_ENTRY.pack(4672, 12, np.float32(0.75)))
                output.write(JOINT_POLICY_ENTRY.pack(8, 4672, np.float32(0.25)))


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


def test_split_conversion_includes_deterministic_replay_files_in_training_only(tmp_path):
    input_dir = tmp_path / "iteration-3" / "training_data"
    replay_dir = tmp_path / "iteration-2" / "training_data"
    input_dir.mkdir(parents=True)
    replay_dir.mkdir(parents=True)

    _write_chunk(input_dir / "chunk_current_000000.hvm", range(20))
    for index in range(10):
        _write_chunk(
            replay_dir / f"chunk_replay_{index:06d}.hvm",
            [100 + index],
        )

    output_dir = tmp_path / "rl_data"
    train_dir, val_dir, _, _ = convert_to_split_parquet(
        input_dir,
        output_dir,
        validation_fraction=0.5,
        split_seed=7,
        samples_per_shard=64,
        replay_input_dir=replay_dir,
        replay_files=2,
        replay_selection_fraction=0.5,
    )

    train_game_ids = set(pl.read_parquet(train_dir / "*.parquet")["game_id"])
    val_game_ids = set(pl.read_parquet(val_dir / "*.parquet")["game_id"])
    replay_game_ids = train_game_ids & set(range(100, 110))
    assert len(replay_game_ids) == 2
    assert not (val_game_ids & set(range(100, 110)))

    convert_to_split_parquet(
        input_dir,
        output_dir,
        validation_fraction=0.5,
        split_seed=7,
        samples_per_shard=64,
        replay_input_dir=replay_dir,
        replay_files=2,
        replay_selection_fraction=0.5,
    )
    repeated_train_ids = set(
        pl.read_parquet(train_dir / "*.parquet")["game_id"]
    )
    assert repeated_train_ids == train_game_ids


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


def test_default_validation_fraction_is_ten_percent():
    assert DEFAULT_RL_VALIDATION_FRACTION == 0.10


def test_default_rl_shards_are_bounded_for_dense_policy_memory():
    assert DEFAULT_RL_SAMPLES_PER_SHARD == 4096


def test_select_replay_files_uses_all_when_replay_files_is_none_or_all(tmp_path):
    from src.preprocessing.convert_selfplay_data import select_replay_files

    replay_dir = tmp_path / "replay_data"
    replay_dir.mkdir()
    for index in range(5):
        _write_chunk(replay_dir / f"chunk_{index:06d}.hvm", [index])

    # replay_files=None -> returns all chunks
    all_chunks = select_replay_files(replay_dir, replay_files=None, replay_selection_fraction=0.05, seed=42)
    assert len(all_chunks) == 5

    # replay_files >= available -> returns all chunks without error
    all_chunks_capped = select_replay_files(replay_dir, replay_files=25, replay_selection_fraction=0.05, seed=42)
    assert len(all_chunks_capped) == 5


def test_model_artifacts_use_separate_directories():
    assert RL_WEIGHTS_DIR.name == "rl"
    assert SUPERVISED_WEIGHTS_DIR.name == "supervised"
    assert RL_WEIGHTS_DIR.parent == SUPERVISED_WEIGHTS_DIR.parent


def test_final_rl_checkpoint_is_resumable(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    checkpoint_path = save_final_rl_checkpoint(model, optimizer, tmp_path, model_prefix="hivemind-rl-it01-risev33")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    assert checkpoint_path.name == "hivemind-rl-it01-risev33.tar"
    assert set(checkpoint) == {"model_state_dict", "optimizer_state_dict"}


def test_soft_cross_entropy_honors_sample_weights():
    from src.training.trainer_agent import SoftCrossEntropyLoss, SampleWeightedLoss

    loss_fn = SampleWeightedLoss(SoftCrossEntropyLoss)
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    weights = torch.tensor([1.0, 0.0])

    loss = loss_fn(logits, targets, weights)
    # The second sample has weight 0.0, so only the first (correct prediction) should contribute
    assert loss.item() < 0.01


def test_joint_policy_loss_learns_pair_correlation_and_pass_id():
    from src.training.trainer_agent import joint_policy_cross_entropy

    rank = 2
    vocabulary = 4673
    policy_a = torch.zeros(1, 4672, requires_grad=True)
    policy_b = torch.zeros(1, 4672, requires_grad=True)
    factors_a = torch.randn(1, rank * vocabulary, requires_grad=True)
    factors_b = torch.randn(1, rank * vocabulary, requires_grad=True)
    target_a = torch.zeros(1, 256, dtype=torch.long)
    target_b = torch.zeros(1, 256, dtype=torch.long)
    probability = torch.zeros(1, 256)
    target_a[0, :2] = torch.tensor([4672, 10])
    target_b[0, :2] = torch.tensor([12, 4672])
    probability[0, :2] = torch.tensor([0.75, 0.25])

    loss = joint_policy_cross_entropy(
        policy_a, policy_b, factors_a, factors_b,
        target_a, target_b, probability, torch.tensor([2]),
        rank=rank, top_k=8,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert factors_a.grad is not None
    assert factors_b.grad is not None
    assert factors_a.grad.abs().sum() > 0


def test_q_value_blending_in_shard_reading(tmp_path):
    from src.preprocessing.convert_selfplay_data import read_binary_shard

    chunk_v4 = tmp_path / "chunk_v4.hvm"
    _write_chunk(chunk_v4, [1], version=4)
    # in _write_chunk: outcome = 1, root_q = 0.5
    # with q_value_ratio = 0.15: blended = 0.85 * 1.0 + 0.15 * 0.5 = 0.925
    samples = read_binary_shard(chunk_v4, q_value_ratio=0.15)
    assert len(samples) == 1    
    assert np.isclose(samples[0]["y_value"], 0.925)
    assert np.isclose(samples[0]["root_q"], 0.5)
    assert np.isclose(samples[0]["outcome"], 1.0)

    chunk_v3 = tmp_path / "chunk_v3.hvm"
    _write_chunk(chunk_v3, [2], version=3)
    samples_v3 = read_binary_shard(chunk_v3, q_value_ratio=0.15)
    assert len(samples_v3) == 1
    assert np.isclose(samples_v3[0]["y_value"], 1.0)
    assert np.isclose(samples_v3[0]["root_q"], 1.0)


def test_hvm5_preserves_sparse_joint_policy_and_distinct_pass(tmp_path):
    from src.preprocessing.convert_selfplay_data import read_binary_shard

    chunk = tmp_path / "chunk_v5.hvm"
    _write_chunk(chunk, [7], version=5)
    sample = read_binary_shard(chunk)[0]

    indices_a = np.frombuffer(sample["joint_policy_a"], dtype=np.uint16)
    indices_b = np.frombuffer(sample["joint_policy_b"], dtype=np.uint16)
    probabilities = np.frombuffer(
        sample["joint_policy_probability"], dtype=np.float32
    )
    assert len(indices_a) == MAX_JOINT_POLICY_ENTRIES
    assert sample["joint_policy_count"] == 2
    assert (indices_a[0], indices_b[0]) == (4672, 12)
    assert np.allclose(probabilities[:2], [0.75, 0.25])
