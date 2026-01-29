"""Cleanup helper for run/output folders (defaults to wiping _runs)."""

import argparse
import os
import shutil
from pathlib import Path


DEFAULT_TARGETS = [
    "_runs",
]


def _dir_stats(path: Path) -> tuple[int, int]:
    """
    Return total bytes and file count for a directory tree.

    @param path: Directory path.
    @return: (total_bytes, total_files)
    """
    total_bytes = 0
    total_files = 0
    for root, _, files in os.walk(path):
        for name in files:
            p = Path(root) / name
            try:
                total_bytes += p.stat().st_size
                total_files += 1
            except OSError:
                pass
    return total_bytes, total_files


def _human_bytes(n: int) -> str:
    """
    Format bytes as a human-readable string.

    @param n: Byte count.
    @return: Human-readable size string.
    """
    unit = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for u in unit:
        if size < 1024.0:
            return f"{size:.2f}{u}"
        size /= 1024.0
    return f"{size:.2f}PB"


def main() -> int:
    """
    Parse args and delete selected folders.

    @return: Exit code.
    """
    ap = argparse.ArgumentParser(description="Cleanup run artifacts and large output folders.")
    ap.add_argument("--yes", action="store_true", help="Actually delete files and folders.")
    ap.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS, help="Folders to remove.")
    args = ap.parse_args()

    root = Path.cwd()
    to_remove = [root / t for t in args.targets]

    print("Cleanup targets:")
    for p in to_remove:
        if p.exists():
            size, files = _dir_stats(p)
            print(f"  - {p} ({files} files, {_human_bytes(size)})")
        else:
            print(f"  - {p} (missing)")

    if not args.yes:
        print("Dry-run only. Re-run with --yes to delete.")
        return 0

    for p in to_remove:
        if p.exists():
            print(f"Deleting {p} ...")
            shutil.rmtree(p, ignore_errors=True)

    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
