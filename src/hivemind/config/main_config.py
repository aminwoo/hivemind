"""Project-relative paths and export settings, adapted from CrazyAra (queensgambit)."""

from hivemind.paths import PROJECT_ROOT as project_root

default_dir = str(project_root / "data") + "/"
phase = None  # current phase to use, set to None to treat everything as a single phase
phase_definition = "movecount3"

main_config = {
    "phase": phase,
    "phase_definition": phase_definition,
    "default_dir": default_dir,

    "pgn_train_dir": default_dir + "pgn/train/",
    "pgn_val_dir": default_dir + "pgn/val/",
    "pgn_test_dir": default_dir + "pgn/test/",
    "pgn_mate_in_one_dir": default_dir + "pgn/mate_in_one/",

    "planes_train_dir": default_dir + "planes/train/",
    "planes_val_dir": default_dir + "planes/val/",
    "planes_test_dir": default_dir + f"planes/{phase_definition}/phase{phase}/test/",
    "planes_mate_in_one_dir": default_dir + f"planes/{phase_definition}/phase{phase}/mate_in_one/",

    "rec_dir": default_dir + "rec/",
    "model_architecture_dir": "/DeepCrazyhouse/models/Classic/symbol/",

    "model_weights_dir": "/DeepCrazyhouse/models/Classic/params/",

    "value_output": "value_out",
    "policy_output": "policy_out",
    "auxiliary_output": "auxiliary_out",
    "wdl_output": "wdl_out",
    "plys_to_end_output": "plys_to_end_out",

    "mode": 2,
    "version": 3,
}
