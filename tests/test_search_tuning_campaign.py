"""Check opening independence and the held-out finalist selection protocol."""
import importlib.util
import json
import struct
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "engine/scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("tuning_campaign", SCRIPTS / "tune_bughouse_search.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


@pytest.mark.parametrize("game_count,pair_counts", [(1064, (64, 200, 800)), (3000, None)])
def test_books_sample_one_position_per_game_and_use_disjoint_games(tmp_path, game_count, pair_counts):
    source = tmp_path / "source"
    source.mkdir()
    plies = [0, 6, 12] * game_count
    count = len(plies)
    chunk = source / "games.bin"
    chunk.write_bytes(struct.pack("<4sIIQQ", b"HNUE", 1, 123, count, 0)
                      + bytes(count * 17) + struct.pack(f"<{count}H", *plies))
    chunk.with_suffix(".fen").write_text("\n".join(
        f"game-{game}-ply-{ply};board-b;w;0" for game in range(game_count) for ply in [0, 6, 12]) + "\n")
    output = tmp_path / "books"
    manifest = campaign.prepare_books(source, output, 42, *(pair_counts or ()))
    expected = pair_counts or (1000, 1000, 1000)
    assert manifest["counts"] == dict(zip(["screen", "validate", "holdout"], expected))
    games = []
    for stage in manifest["counts"]:
        lines = (output / f"{stage}.tsv").read_text().splitlines()
        games.extend(line.split("-ply-")[0] for line in lines)
        assert all("-ply-6|" in line or "-ply-12|" in line for line in lines)
    assert len(games) == len(set(games)) == game_count


def test_finalist_is_selected_on_validation_before_held_out_test(tmp_path):
    engine, model = tmp_path / "engine", tmp_path / "net.onnx"
    engine.write_bytes(b"binary")
    model.write_bytes(b"network")
    output = tmp_path / "campaign"
    books = output / "openings"
    books.mkdir(parents=True)
    for stage in ["screen", "validate", "holdout"]:
        (books / f"{stage}.tsv").write_text(stage + "\n")
    (books / "openings.json").write_text(json.dumps({
        "counts": {stage: 1 for stage in ["screen", "validate", "holdout"]},
        "sha256": {stage: campaign.file_digest(books / f"{stage}.tsv")
                   for stage in ["screen", "validate", "holdout"]}}))
    runs = []

    def tournament(command, **kwargs):
        directory = Path(command[command.index("--output") + 1])
        stage, name = directory.parent.name, directory.name
        runs.append((stage, name, Path(command[command.index("--positions") + 1]).name))
        score = {("screen", "alpha"): .9, ("screen", "beta"): .7,
                 ("validate", "alpha"): .4, ("validate", "beta"): .6}.get((stage, name), .5)
        (directory / "sweep.json").write_text(json.dumps({"runs": [{
            "games": 2, "contender_score": score, "contender_elo": 0,
            "elo_confidence_95": [-50, 50], "score_confidence_95": [.4, .6]}]}))

    args = ["tune", "--engine", str(engine), "--model", str(model), "--output", str(output),
            "--screen-games", "2", "--validate-games", "2", "--holdout-games", "2"]
    candidates = {"control": {"pw-mass": 0}, "alpha": {"cpuct-init": 2}, "beta": {"cpuct-init": 4}}
    with patch.object(sys, "argv", args), patch.object(campaign, "CANDIDATES", candidates), \
            patch.object(campaign.subprocess, "run", tournament):
        assert campaign.main() == 0
    assert [row for row in runs if row[0] == "holdout"] == [("holdout", "beta", "holdout.tsv")]
    assert all(book == f"{stage}.tsv" for stage, _, book in runs)
    state = json.loads((output / "campaign.json").read_text())
    assert state["status"] == "complete"
    assert "No statistically established gain" in state["conclusion"]
