"""Download back the files `hivemind archive` or `hivemind rl-loop` moved from
each DIR to Hugging Face, as DIR/archive.json lists them. Files already on
disk are left alone."""

import argparse
from pathlib import Path
import sys

from hivemind.data import archive


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="+", type=Path, metavar="DIR")
    args = parser.parse_args()
    unarchived = [str(directory) for directory in args.directories
                  if not (directory / archive.MANIFEST).is_file()]
    if unarchived:
        parser.error(f"no {archive.MANIFEST} in {', '.join(unarchived)}")
    for directory in args.directories:
        print(f"{directory}: restored {archive.restore(directory)} files", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
