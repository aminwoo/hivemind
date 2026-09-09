#!/usr/bin/env python3
"""Bootstrap the network download without installing the Python package."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hivemind.cli.fetch_network import main

if __name__ == "__main__":
    main()
