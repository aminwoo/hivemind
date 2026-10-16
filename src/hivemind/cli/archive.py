"""Move data directories to a private Hugging Face dataset, freeing the disk.

Each DIR's files are packed into a zstd tarball, read back against the files,
uploaded to --repo and checked against the Hub's copy before they are deleted.
DIR keeps an archive.json listing what went where, and `hivemind restore DIR`
brings the files back. Running this again on DIR uploads only the files added
or changed since. Don't point it at a directory something is still writing to:
for the RL loop's runs use `hivemind rl-loop --archive-only`, which leaves
unfinished self-play alone."""

import argparse
from pathlib import Path
import sys

from hivemind.data import archive
from hivemind.network import SELFPLAY_ARCHIVE


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="+", type=Path, metavar="DIR")
    parser.add_argument("--repo", default=SELFPLAY_ARCHIVE,
                        help="Private Hugging Face dataset to move the files to (created if missing)")
    args = parser.parse_args()
    missing = [str(directory) for directory in args.directories if not directory.is_dir()]
    if missing:
        parser.error(f"not a directory: {', '.join(missing)}")
    from huggingface_hub import HfApi
    api = HfApi()
    for directory in args.directories:
        freed = archive.archive(api, args.repo, directory)
        print(f"{directory}: freed {freed / 1e9:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
