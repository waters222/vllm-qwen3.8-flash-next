"""Export an allowlisted source-only Docker context; never include the checkout."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def export(root, output):
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("Build output must be an existing empty directory")
    manifest = json.loads((root / "docs/flash_next/source_manifest.json").read_text())
    revisions = manifest.get("post_snapshot_sources", {})
    if not revisions.keys() <= manifest["engine_sources"].keys():
        raise ValueError("Source revisions cannot expand the engine allowlist")
    for name, expected in manifest["engine_sources"].items():
        source = root / name
        current_expected = revisions.get(name, expected)
        if hashlib.sha256(source.read_bytes()).hexdigest() != current_expected:
            raise ValueError("Engine source differs: " + name)
        target = output / "engine" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # The builder carries the exact upstream C++ source pins.
    import build_flash_marlin_schedule as build

    base = build.load_base()
    for name, expected in base.PINS.items():
        source = root / "csrc" / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError("CUDA source differs: " + name)
        target = output / "csrc" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in ("build_flash_marlin_schedule.py", "build_flash_marlin_pipeline.py"):
        target = output / "builder" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "tools/flash_next" / name, target)
    shutil.copyfile(root / "tools/flash_next/Dockerfile", output / "Dockerfile")
    return len(manifest["engine_sources"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = export(Path(__file__).resolve().parents[2], args.output.resolve())
    print(f"Exported {count} pinned engine files and the CUDA build inputs")


if __name__ == "__main__":
    main()
