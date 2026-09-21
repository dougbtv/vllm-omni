# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Verify runtime package data in a built vLLM-Omni wheel.

The source tree may contain package data through setuptools-scm's Git file
finder even when a VCS-less source archive cannot rediscover it. This check
compares each runtime-owned source directory with both the wheel payload and,
optionally, an installed copy of that wheel.
"""

import argparse
import sys
import zipfile
from pathlib import Path

PACKAGE_NAME = "vllm_omni"
RUNTIME_DATA_DIRS = (
    Path("deploy"),
    Path("diffusion/distributed/csrc"),
    Path("model_executor/models/covo_audio/speaker_prompt"),
    Path("model_executor/models/step_audio2/assets"),
    Path("transformers_utils/chat_templates"),
)


def runtime_data_files(source_root: Path) -> list[Path]:
    package_root = source_root / PACKAGE_NAME
    files: list[Path] = []
    for relative_dir in RUNTIME_DATA_DIRS:
        data_dir = package_root / relative_dir
        if not data_dir.is_dir():
            raise FileNotFoundError(f"runtime data directory not found: {data_dir}")
        files.extend(
            path.relative_to(source_root) for path in data_dir.rglob("*") if path.is_file() and path.suffix != ".py"
        )
    return sorted(files)


def missing_wheel_files(wheel: Path, expected: list[Path]) -> list[Path]:
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
    return [path for path in expected if path.as_posix() not in members]


def missing_installed_files(installed_root: Path, expected: list[Path]) -> list[Path]:
    missing = []
    for path in expected:
        installed_path = installed_root / path
        if not installed_path.is_file():
            missing.append(path)
            continue
        installed_path.read_bytes()
    return missing


def report_missing(label: str, missing: list[Path]) -> bool:
    if not missing:
        return False
    print(f"{label} is missing {len(missing)} runtime package-data file(s):", file=sys.stderr)
    for path in missing:
        print(f"  {path.as_posix()}", file=sys.stderr)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--installed-root", type=Path)
    args = parser.parse_args()

    expected = runtime_data_files(args.source_root.resolve())
    failed = report_missing("wheel", missing_wheel_files(args.wheel, expected))
    if args.installed_root is not None:
        failed |= report_missing(
            "installed wheel",
            missing_installed_files(args.installed_root.resolve(), expected),
        )

    if failed:
        return 1
    print(f"validated {len(expected)} runtime package-data files in {args.wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
