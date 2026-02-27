#!/usr/bin/env python3
"""Batch-convert HEIC/HEIF photos to PNG for analysis/paper use.

Typical use:
  python tools/convert_heic_to_png.py --input data/Real_World --recursive

Notes:
  - Requires pillow-heif for HEIC/HEIF decoding:
      pip install pillow-heif
  - Keeps original files unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image


HEIF_EXTS = {".heic", ".heif"}


def _register_heif() -> None:
    try:
        import pillow_heif  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "HEIC/HEIF decoding requires pillow-heif. Install with: pip install pillow-heif"
        ) from e

    if hasattr(pillow_heif, "register_heif_opener"):
        pillow_heif.register_heif_opener()
        return
    raise RuntimeError(
        "pillow-heif is installed, but register_heif_opener() was not found."
    )


def _iter_heif_files(inp: Path, recursive: bool) -> list[Path]:
    if inp.is_file():
        if inp.suffix.lower() not in HEIF_EXTS:
            raise ValueError(f"Input file is not HEIC/HEIF: {inp}")
        return [inp]
    if not inp.exists():
        raise FileNotFoundError(f"Input path not found: {inp}")
    if not inp.is_dir():
        raise ValueError(f"Input must be a file or directory: {inp}")
    it: Iterable[Path] = inp.rglob("*") if recursive else inp.iterdir()
    files = [p for p in it if p.is_file() and p.suffix.lower() in HEIF_EXTS]
    return sorted(files)


def _default_output_path(src: Path, input_root: Path, out_dir: Path | None) -> Path:
    if out_dir is None:
        return src.with_suffix(".png")
    try:
        rel = src.relative_to(input_root)
    except Exception:
        rel = src.name
    if isinstance(rel, Path):
        return (out_dir / rel).with_suffix(".png")
    return (out_dir / str(rel)).with_suffix(".png")


def convert_one(src: Path, dst: Path, overwrite: bool, png_compress_level: int) -> str:
    if dst.exists() and not overwrite:
        return "skip_exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        # Preserve alpha if present; otherwise RGB.
        if im.mode not in ("RGB", "RGBA"):
            if "A" in im.mode:
                im = im.convert("RGBA")
            else:
                im = im.convert("RGB")
        im.save(dst, format="PNG", compress_level=png_compress_level)
    return "converted"


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert HEIC/HEIF photos to PNG.")
    ap.add_argument("--input", required=True, help="HEIC file or folder containing HEIC/HEIF files")
    ap.add_argument("--recursive", action="store_true", help="Recursively scan folder input")
    ap.add_argument("--out-dir", default="", help="Optional output directory (preserves relative paths)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files")
    ap.add_argument(
        "--compress-level",
        type=int,
        default=6,
        choices=list(range(10)),
        help="PNG compression level (0-9). Default 6",
    )
    args = ap.parse_args()

    _register_heif()

    inp = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else None
    files = _iter_heif_files(inp, recursive=bool(args.recursive))
    if not files:
        raise FileNotFoundError(f"No HEIC/HEIF files found under: {inp}")

    print(f"[CONVERT] found={len(files)} input={inp}")
    if out_dir:
        print(f"[CONVERT] out_dir={out_dir}")

    converted = 0
    skipped = 0
    failed = 0

    input_root = inp if inp.is_dir() else inp.parent

    for src in files:
        dst = _default_output_path(src, input_root=input_root, out_dir=out_dir)
        try:
            status = convert_one(src, dst, overwrite=bool(args.overwrite), png_compress_level=int(args.compress_level))
            if status == "converted":
                converted += 1
                print(f"[OK]   {src.name} -> {dst}")
            else:
                skipped += 1
                print(f"[SKIP] {dst} exists")
        except Exception as e:
            failed += 1
            print(f"[ERR]  {src} :: {e}")

    print(f"[DONE] converted={converted} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

