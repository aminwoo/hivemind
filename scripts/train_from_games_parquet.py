"""
End-to-end supervised training pipeline from a games parquet file.

This script performs:
1) Generate training plane shards from games.parquet
2) Build a validation shard (evaluation_shard.parquet) if needed
3) Launch supervised training
4) Produce checkpoints in src/training/weights/supervised

Example:
    python scripts/train_from_games_parquet.py --games data/games.parquet
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.generate_planes import generate_planes


DATASET_META_FILE = "dataset_meta.json"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clear_parquet_files(folder: Path) -> None:
    if not folder.exists():
        return
    for file_path in folder.glob("*.parquet"):
        file_path.unlink()


def _validate_parquet_files(file_paths: list[Path]) -> None:
    for file_path in file_paths:
        try:
            pl.read_parquet_schema(file_path)
        except Exception as error:
            raise RuntimeError(f"Invalid parquet file: {file_path}: {error}") from error


def _dataset_meta_path(train_dir: Path) -> Path:
    return train_dir / DATASET_META_FILE


def _write_dataset_metadata(
    train_dir: Path,
    games_path: Path,
    min_rating: int,
    augment_board_swap: bool,
    val_fraction: float,
    seed: int,
) -> None:
    meta = {
        "split_version": 1,
        "games_path": str(games_path),
        "min_rating": int(min_rating),
        "augment_board_swap": bool(augment_board_swap),
        "val_fraction": float(val_fraction),
        "seed": int(seed),
    }
    _dataset_meta_path(train_dir).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def _read_dataset_metadata(train_dir: Path) -> dict | None:
    path = _dataset_meta_path(train_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_games_path(games_arg: str, project_root: Path) -> Path:
    candidate = Path(games_arg)
    if candidate.is_absolute():
        return candidate

    # 1) Relative to current working directory.
    cwd_candidate = candidate.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    # 2) Relative to repository root.
    root_candidate = (project_root / candidate).resolve()
    if root_candidate.exists():
        return root_candidate

    # 3) Common shorthand: --games games.parquet -> repo/data/games.parquet.
    data_candidate = (project_root / "data" / candidate).resolve()
    if data_candidate.exists():
        return data_candidate

    return cwd_candidate


def _write_parquet_atomic(frame: pl.DataFrame, output_path: Path) -> None:
    _ensure_dir(output_path.parent)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    frame.write_parquet(temporary_path, compression="zstd")
    temporary_path.replace(output_path)


def _build_train_evaluation_shard(
    train_dir: Path,
    output_path: Path,
    max_samples: int,
    seed: int,
) -> int:
    train_files = sorted(train_dir.glob("*.parquet"))
    if not train_files:
        raise FileNotFoundError(f"No training shards found in {train_dir}")

    sampled_frames = []
    samples_per_shard = max(1, (max_samples + len(train_files) - 1) // len(train_files))
    for index, shard in enumerate(train_files):
        df = pl.read_parquet(shard)
        if df.is_empty():
            continue
        sampled_frames.append(
            df.sample(
                n=min(samples_per_shard, df.height),
                seed=seed + index,
                with_replacement=False,
            )
        )

    if not sampled_frames:
        raise RuntimeError("Could not construct train evaluation shard: all shards were empty")

    evaluation_df = pl.concat(sampled_frames, how="vertical")
    if evaluation_df.height > max_samples:
        evaluation_df = evaluation_df.sample(
            n=max_samples,
            seed=seed,
            with_replacement=False,
        )

    _write_parquet_atomic(evaluation_df, output_path)
    return evaluation_df.height


def _generate_validation_shard(
    games_path: Path,
    val_shard_path: Path,
    min_rating: int,
    augment_board_swap: bool,
    val_fraction: float,
    max_val_samples: int,
    seed: int,
) -> int:
    _clear_parquet_files(val_shard_path.parent)
    generated_paths = generate_planes(
        samples_per_shard=max_val_samples,
        games_path=str(games_path),
        output_dir=str(val_shard_path.parent),
        min_rating=min_rating,
        augment_board_swap=augment_board_swap,
        split="val",
        val_fraction=val_fraction,
        seed=seed,
        max_samples=max_val_samples,
    )
    if len(generated_paths) != 1:
        raise RuntimeError(
            f"Expected one validation shard, generated {len(generated_paths)}"
        )

    generated_path = Path(generated_paths[0])
    generated_path.replace(val_shard_path)
    return pl.scan_parquet(val_shard_path).select(pl.len()).collect().item()


def _run_supervised_training(
    project_root: Path,
    train_planes_dir: Path,
    validation_shard: Path,
    architecture: str,
    batch_size: int | None = None,
    precision: str | None = None,
    checkpoint_path: Path | None = None,
    train_eval_shard: Path | None = None,
) -> None:
    train_dir = project_root / "src" / "training"
    _ensure_dir(train_dir / "weights" / "supervised")
    _ensure_dir(train_dir / "logs")

    cmd = [
        sys.executable,
        "train_loop.py",
        "--mode",
        "sl",
        "--architecture",
        architecture,
        "--sl-train-data-dir",
        str(train_planes_dir),
        "--sl-validation-shard",
        str(validation_shard),
    ]
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    if precision is not None:
        cmd.extend(["--precision", precision])
    if checkpoint_path is not None:
        cmd.extend(["--checkpoint", str(checkpoint_path)])
    if train_eval_shard is not None:
        cmd.extend(["--train-eval-shard", str(train_eval_shard)])
    subprocess.run(cmd, cwd=train_dir, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process games.parquet, train supervised model, and export ONNX checkpoints"
    )
    parser.add_argument(
        "--games",
        type=str,
        default=str(PROJECT_ROOT / "data" / "games.parquet"),
        help="Path to input games parquet",
    )
    parser.add_argument(
        "--train-planes-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "planes" / "train"),
        help="Directory for generated training plane shards",
    )
    parser.add_argument(
        "--val-shard",
        type=str,
        default=str(PROJECT_ROOT / "data" / "planes" / "val" / "evaluation_shard.parquet"),
        help="Validation shard parquet output path",
    )
    parser.add_argument(
        "--train-eval-shard",
        type=str,
        default=str(PROJECT_ROOT / "data" / "planes" / "train_eval" / "evaluation_shard.parquet"),
        help="Fixed representative training shard used only for metrics",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=2**16,
        help="Number of samples per generated training shard",
    )
    parser.add_argument(
        "--min-rating",
        "--rating-threshold",
        dest="min_rating",
        type=int,
        default=2200,
        help="Minimum rating threshold. Only games where all 4 players meet this rating are used.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.02,
        help="Fraction of complete paired games reserved for validation",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=2**16,
        help="Cap on total validation samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for validation sampling",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip plane generation and reuse existing train shards",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only generate data, do not launch supervised training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .tar checkpoint from which to resume supervised training",
    )
    parser.add_argument(
        "--architecture",
        choices=(
            "risev33",
            "crossboard-risev33",
            "dualstream-memory-risev33",
        ),
        default="risev33",
        help="Network architecture passed to supervised training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional training batch-size override",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default=None,
        help="Optional model execution precision override",
    )
    parser.add_argument(
        "--rebuild-val",
        action="store_true",
        help="Regenerate validation planes from the held-out game split",
    )
    parser.add_argument(
        "--clean-train-dir",
        action="store_true",
        help="Delete existing train shard parquet files before generation",
    )
    parser.add_argument(
        "--no-board-swap-doubling",
        action="store_false",
        dest="augment_board_swap",
        help="Disable supervised board-swap doubling (enabled by default)",
    )
    parser.set_defaults(augment_board_swap=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    games_path = _resolve_games_path(args.games, PROJECT_ROOT)
    train_planes_dir = Path(args.train_planes_dir).resolve()
    val_shard_path = Path(args.val_shard).resolve()
    train_eval_shard_path = Path(args.train_eval_shard).resolve()
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    )

    if not games_path.exists() and not args.skip_generate:
        raise FileNotFoundError(f"games parquet not found: {games_path}")
    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _ensure_dir(train_planes_dir)
    _ensure_dir(val_shard_path.parent)
    _ensure_dir(train_eval_shard_path.parent)

    if not args.skip_generate:
        existing_train_files = list(train_planes_dir.glob("*.parquet"))
        if existing_train_files and not args.clean_train_dir:
            raise RuntimeError(
                f"Training shards already exist in {train_planes_dir}. "
                "Pass --clean-train-dir to replace them with the leakage-free split."
            )
        if args.clean_train_dir:
            _clear_parquet_files(train_planes_dir)

        print(
            f"Generating train plane shards from {games_path} "
            f"with rating threshold >= {args.min_rating}; "
            f"board-swap doubling={'on' if args.augment_board_swap else 'off'}"
        )
        generate_planes(
            samples_per_shard=args.samples_per_shard,
            games_path=str(games_path),
            output_dir=str(train_planes_dir),
            min_rating=args.min_rating,
            augment_board_swap=args.augment_board_swap,
            split="train",
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        print(f"Generating held-out validation shard at {val_shard_path}")
        n_val = _generate_validation_shard(
            games_path=games_path,
            val_shard_path=val_shard_path,
            min_rating=args.min_rating,
            augment_board_swap=args.augment_board_swap,
            val_fraction=args.val_fraction,
            max_val_samples=args.max_val_samples,
            seed=args.seed,
        )
        print(f"Validation samples written: {n_val}")
        _write_dataset_metadata(
            train_planes_dir,
            games_path,
            args.min_rating,
            args.augment_board_swap,
            args.val_fraction,
            args.seed,
        )
    else:
        meta = _read_dataset_metadata(train_planes_dir)
        if meta is None:
            raise RuntimeError(
                f"Cannot verify rating threshold for existing shards in {train_planes_dir}. "
                f"Missing {DATASET_META_FILE}. Regenerate shards without --skip-generate."
            )

        if int(meta.get("split_version", 0)) != 1:
            raise RuntimeError(
                "Existing shards predate whole-game validation splitting. "
                "Regenerate with --clean-train-dir before resuming training."
            )

        existing_min_rating = int(meta.get("min_rating", -1))
        existing_board_swap = bool(meta.get("augment_board_swap", False))
        if existing_min_rating < args.min_rating:
            raise RuntimeError(
                "Existing train shards were generated with a lower rating threshold "
                f"({existing_min_rating}) than requested ({args.min_rating}). "
                "Regenerate shards without --skip-generate."
            )
        if args.augment_board_swap and not existing_board_swap:
            raise RuntimeError(
                "Existing train shards were generated without supervised board-swap doubling. "
                "Regenerate shards without --skip-generate or pass --no-board-swap-doubling."
            )
        if float(meta.get("val_fraction", -1)) != args.val_fraction:
            raise RuntimeError(
                "Existing shards use a different validation fraction. Regenerate them."
            )
        if int(meta.get("seed", -1)) != args.seed:
            raise RuntimeError(
                "Existing shards use a different split seed. Regenerate them."
            )

        print(
            f"Reusing existing train shards filtered with rating threshold >= {existing_min_rating}; "
            f"board-swap doubling={'on' if existing_board_swap else 'off'}"
        )

    train_files = sorted(train_planes_dir.glob("*.parquet"))
    if not train_files:
        raise RuntimeError(
            f"No train shard parquet files found in {train_planes_dir}. "
            "Generation may have failed or all games were filtered out."
        )
    _validate_parquet_files(train_files)
    print(f"Train shards available: {len(train_files)}")

    if args.skip_generate and args.rebuild_val:
        if not games_path.is_file():
            raise FileNotFoundError(f"games parquet not found: {games_path}")
        print(f"Regenerating held-out validation shard at {val_shard_path}")
        n_val = _generate_validation_shard(
            games_path=games_path,
            val_shard_path=val_shard_path,
            min_rating=args.min_rating,
            augment_board_swap=args.augment_board_swap,
            val_fraction=args.val_fraction,
            max_val_samples=args.max_val_samples,
            seed=args.seed,
        )
        print(f"Validation samples written: {n_val}")

    if not val_shard_path.is_file():
        raise FileNotFoundError(f"Validation shard not found: {val_shard_path}")
    _validate_parquet_files([val_shard_path])
    print(f"Using held-out validation shard: {val_shard_path}")

    print(f"Building fixed training evaluation shard at {train_eval_shard_path}")
    n_train_eval = _build_train_evaluation_shard(
        train_dir=train_planes_dir,
        output_path=train_eval_shard_path,
        max_samples=args.max_val_samples,
        seed=args.seed,
    )
    print(f"Training evaluation samples written: {n_train_eval}")

    if args.skip_train:
        print("Skipping training by request (--skip-train).")
        return

    print("Starting supervised training...")
    _run_supervised_training(
        PROJECT_ROOT,
        train_planes_dir=train_planes_dir,
        validation_shard=val_shard_path,
        architecture=args.architecture,
        batch_size=args.batch_size,
        precision=args.precision,
        checkpoint_path=checkpoint_path,
        train_eval_shard=train_eval_shard_path,
    )

    supervised_weights = PROJECT_ROOT / "src" / "training" / "weights" / "supervised"
    onnx_files = sorted(supervised_weights.glob("*.onnx"))
    if onnx_files:
        print("ONNX checkpoints:")
        for onnx_path in onnx_files[-10:]:
            print(f"  - {onnx_path}")
    else:
        print(f"No ONNX files found yet in {supervised_weights}")


if __name__ == "__main__":
    main()
