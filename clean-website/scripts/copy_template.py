#!/usr/bin/env python3
"""Copy the clean website template into a target directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy clean website template files.")
    parser.add_argument("target", help="Directory to receive index.html and style.css")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skill_root = Path(__file__).resolve().parents[1]
    template_dir = skill_root / "assets" / "template"
    target_dir = Path(args.target).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    for name in ("index.html", "style.css"):
        source = template_dir / name
        destination = target_dir / name
        if destination.exists() and not args.force:
            raise FileExistsError(f"{destination} already exists; pass --force to overwrite")
        shutil.copy2(source, destination)

    print(f"Copied clean website template to {target_dir}")


if __name__ == "__main__":
    main()
