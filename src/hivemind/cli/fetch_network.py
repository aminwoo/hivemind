#!/usr/bin/env python3
"""Download a pinned Hivemind network, verifying SHA-256 before installation."""

import argparse
import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen

from hivemind.network import ARTIFACTS, MODEL_DIRECTORY, MODEL_REPOSITORY, MODEL_REVISION


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_network(variant="onnx", output_dir=MODEL_DIRECTORY):
    filename, expected_digest = ARTIFACTS[variant]
    destination = Path(output_dir) / filename
    if destination.is_file() and file_digest(destination) == expected_digest:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/{filename}"
    temporary = None
    try:
        with urlopen(url, timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                temporary = Path(output.name)
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if file_digest(temporary) != expected_digest:
            raise ValueError(f"SHA-256 mismatch for {filename}; download was not installed")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=ARTIFACTS, default="onnx")
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIRECTORY)
    args = parser.parse_args()
    print(fetch_network(args.variant, args.output_dir))


if __name__ == "__main__":
    main()
