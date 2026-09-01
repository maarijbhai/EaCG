#!/usr/bin/env python3
"""Check the local PTB-XL download against the published SHA256SUMS.

Why this exists. The dataset is fetched over HTTP as ~44,000 individual
files, and a bulk fetch that drops a few hundred of them fails *silently*:
every remaining file is valid, the metadata CSV still lists all 21,799
records, and a loader that skips missing files will happily train on the
subset. The damage is subtle rather than fatal -- results stop being
comparable with published PTB-XL benchmarks, and if the losses are not
uniform across `strat_fold` the train/test folds are no longer the
published ones either.

Existence alone is not enough: a connection dropped mid-file leaves a
short, readable, wrong `.dat`. Only the checksums catch that.

Usage::

    python python/verify_ptbxl.py                  # full checksum pass
    python python/verify_ptbxl.py --quick          # existence only, fast
    python python/verify_ptbxl.py --rate 100       # one sampling rate
    python python/verify_ptbxl.py --write-missing missing.txt

Exit status is 0 only if every expected file is present and correct, so it
is safe to gate an experiment run on it.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "ptb-xl" / "1.0.3"

CHUNK = 1 << 20  # 1 MiB


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sums(root: Path) -> list[tuple[str, str]]:
    """Parse SHA256SUMS.txt into [(expected_hex, relative_path), ...]."""
    sums_path = root / "SHA256SUMS.txt"
    if not sums_path.exists():
        raise SystemExit(f"no SHA256SUMS.txt under {root}")
    entries = []
    for line in sums_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        # "<64 hex>  <path>" -- the separator is whitespace, and paths may
        # themselves contain spaces, so split once only.
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        entries.append((parts[0], parts[1].lstrip("*").strip()))
    return entries


def wanted(relative: str, rate: str) -> bool:
    if rate == "both":
        return True
    if relative.startswith("records100/"):
        return rate == "100"
    if relative.startswith("records500/"):
        return rate == "500"
    return True  # metadata files are always wanted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--rate", choices=("100", "500", "both"), default="both")
    parser.add_argument("--quick", action="store_true",
                        help="check existence and size only, skip hashing")
    parser.add_argument("--write-missing", type=Path, default=None,
                        help="write missing/corrupt relative paths, one per line")
    args = parser.parse_args(argv)

    root: Path = args.root
    if not root.exists():
        print(f"dataset root does not exist: {root}", file=sys.stderr)
        return 2

    entries = [(h, p) for h, p in read_sums(root) if wanted(p, args.rate)]
    print(f"root      : {root}")
    print(f"expecting : {len(entries)} files (rate={args.rate})")
    print(f"mode      : {'existence only' if args.quick else 'full sha256'}")

    missing: list[str] = []
    corrupt: list[str] = []
    checked = 0

    for expected, relative in entries:
        path = root / relative
        if not path.exists():
            missing.append(relative)
            continue
        if not args.quick:
            if sha256_of(path) != expected:
                corrupt.append(relative)
        checked += 1
        if checked % 5000 == 0:
            print(f"  ... {checked}/{len(entries)}", flush=True)

    ok = len(entries) - len(missing) - len(corrupt)
    print()
    print(f"ok      : {ok}")
    print(f"missing : {len(missing)}")
    print(f"corrupt : {len(corrupt)}" + ("  (skipped: --quick)" if args.quick else ""))

    for label, items in (("missing", missing), ("corrupt", corrupt)):
        if items:
            print(f"\nfirst {min(10, len(items))} {label}:")
            for item in items[:10]:
                print(f"  {item}")

    if args.write_missing and (missing or corrupt):
        args.write_missing.write_text(
            "\n".join(missing + corrupt) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {len(missing) + len(corrupt)} paths to {args.write_missing}")

    return 0 if not missing and not corrupt else 1


if __name__ == "__main__":
    raise SystemExit(main())
