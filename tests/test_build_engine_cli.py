"""Tests for the C++ engine build command."""

import sys

from hivemind.cli import build_engine


def test_build_engine_uses_requested_preset_and_target(monkeypatch, tmp_path):
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    (engine_dir / "CMakeLists.txt").touch()
    calls = []

    monkeypatch.setattr(build_engine, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        build_engine.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["hivemind build-engine", "--preset", "ninja-release", "--jobs", "3"],
    )

    assert build_engine.main() == 0
    assert calls == [
        (
            ["cmake", "--preset", "ninja-release"],
            {"cwd": engine_dir, "check": True},
        ),
        (
            [
                "cmake", "--build", "--preset", "ninja-release",
                "--target", "hivemind", "--parallel", "3",
            ],
            {"cwd": engine_dir, "check": True},
        ),
    ]
