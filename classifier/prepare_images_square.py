"""Prepare classifier/images_square by extracting waste-pictures.zip.

Usage:
  python3 classifier/prepare_images_square.py \
      --zip /home/triuq/projects/detect-waste/waste-pictures.zip
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create classifier/images_square and unzip waste-pictures.")
    parser.add_argument(
        "--zip",
        dest="zip_path",
        required=True,
        help="Path to waste-pictures.zip",
    )
    parser.add_argument(
        "--dst",
        default="classifier/images_square",
        help="Destination directory (default: classifier/images_square)",
    )
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="Remove destination directory first if it exists.",
    )
    return parser


def ensure_dir(dst: Path, force_clean: bool) -> None:
    if dst.exists() and force_clean:
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)


def validate_structure(dst: Path) -> None:
    train_dir = dst / "train"
    test_dir = dst / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise RuntimeError(
            "Unzip finished but expected train/test folders were not found under "
            f"{dst}."
        )


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    zip_path = Path(args.zip_path).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    ensure_dir(dst, args.force_clean)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst)

    validate_structure(dst)

    train_classes = [p for p in (dst / "train").iterdir() if p.is_dir()]
    test_classes = [p for p in (dst / "test").iterdir() if p.is_dir()]

    print(f"Done: extracted to {dst}")
    print(f"Train classes: {len(train_classes)}")
    print(f"Test classes: {len(test_classes)}")
