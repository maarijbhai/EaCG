#!/usr/bin/env python3
"""Resumable, polite fetcher for the parts of PTB-XL not yet on disk.

Not part of the module layout in CLAUDE.md -- added because the dataset
arrives as ~44,000 separate HTTP requests and getting that wrong is how the
local copy ended up incomplete in the first place.

Two things this does that a naive bulk fetch does not:

**It resumes.** The work list is recomputed from what is actually on disk
at the start of every pass, so interrupting it and re-running costs
nothing. Runs until a pass downloads nothing new.

**It stays under the rate limit.** PhysioNet will refuse connections
outright if hammered -- roughly 4,500 requests at 28-way concurrency was
enough to get this machine temporarily blocked, and once blocked, *every*
subsequent connection fails for several minutes, including ones that would
otherwise have succeeded. The default of 4 concurrent requests is
deliberately modest; the full 500 Hz set takes a while at that rate and
that is the correct trade against being cut off entirely.

Verify afterwards -- a truncated file is present, readable and wrong::

    python python/fetch_ptbxl.py --rate 500
    python python/verify_ptbxl.py --rate 500

Usage::

    python python/fetch_ptbxl.py                # everything missing
    python python/fetch_ptbxl.py --rate 100     # one sampling rate
    python python/fetch_ptbxl.py --concurrency 2 --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "ptb-xl" / "1.0.3"
BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_ptbxl import read_sums, wanted  # noqa: E402


def missing_paths(root: Path, rate: str) -> list[str]:
    """Relative paths listed in SHA256SUMS.txt that are not on disk.

    Existence only -- hashing every candidate on every pass would dominate
    the runtime. Run verify_ptbxl.py afterwards to catch truncated files.
    """
    return [p for _, p in read_sums(root) if wanted(p, rate) and not (root / p).exists()]


def fetch_pass(root: Path, paths: list[str], concurrency: int, timeout: int) -> int:
    """Download `paths` with one curl invocation. Returns curl's exit code."""
    config = root / ".fetch_urls.tmp"
    with config.open("w", encoding="utf-8") as handle:
        for relative in paths:
            handle.write(f'url = "{BASE_URL}{relative}"\noutput = "{relative}"\n')
    try:
        completed = subprocess.run(
            ["curl", "-sS", "--parallel", "--parallel-max", str(concurrency),
             "--create-dirs", "--retry", "3", "--retry-delay", "5",
             "--connect-timeout", str(timeout), "-K", config.name],
            cwd=root, capture_output=True, text=True,
        )
        if completed.returncode != 0:
            tail = completed.stderr.strip().splitlines()[-1:] or ["(no stderr)"]
            print(f"    curl exit {completed.returncode}: {tail[0]}")
        return completed.returncode
    finally:
        config.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--rate", choices=("100", "500", "both"), default="both")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="parallel requests; raising this risks a block")
    parser.add_argument("--batch", type=int, default=2000,
                        help="files per curl invocation")
    parser.add_argument("--max-passes", type=int, default=10)
    parser.add_argument("--cooldown", type=int, default=30,
                        help="seconds to wait between passes")
    parser.add_argument("--connect-timeout", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root: Path = args.root
    if not (root / "SHA256SUMS.txt").exists():
        print(f"no SHA256SUMS.txt under {root}", file=sys.stderr)
        return 2

    for attempt in range(1, args.max_passes + 1):
        pending = missing_paths(root, args.rate)
        print(f"pass {attempt}: {len(pending)} files missing")
        if not pending:
            print("nothing left to fetch")
            return 0
        if args.dry_run:
            for relative in pending[:10]:
                print(f"  would fetch {relative}")
            print(f"  ... and {max(0, len(pending) - 10)} more")
            return 0

        for start in range(0, len(pending), args.batch):
            batch = pending[start: start + args.batch]
            print(f"  batch {start // args.batch + 1}: {len(batch)} files", flush=True)
            fetch_pass(root, batch, args.concurrency, args.connect_timeout)

        remaining = len(missing_paths(root, args.rate))
        gained = len(pending) - remaining
        print(f"  fetched {gained}, {remaining} still missing")
        if gained == 0:
            # A pass that gains nothing means the server is refusing us, not
            # that the files are unavailable. Backing off is the only useful
            # response; hammering harder is what caused it.
            print(f"  no progress -- backing off {args.cooldown}s", flush=True)
            time.sleep(args.cooldown)
        if remaining == 0:
            print("complete")
            return 0

    print(f"gave up after {args.max_passes} passes; re-run to continue")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
