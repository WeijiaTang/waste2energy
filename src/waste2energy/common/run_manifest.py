# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

"""Stable run manifests for reproducible SCI companion workflows."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class FileDigest:
    path: str
    exists: bool
    size_bytes: int | None
    sha256: str | None


def file_digest(path: str | Path, *, root: str | Path | None = None) -> FileDigest:
    """Hash a file while preserving missing-file information."""

    root_path = Path(root).resolve() if root is not None else None
    raw_path = Path(path)
    resolved = (root_path / raw_path).resolve() if root_path and not raw_path.is_absolute() else raw_path.resolve()
    display_path = _display_path(resolved, root_path)
    if not resolved.exists() or not resolved.is_file():
        return FileDigest(path=display_path, exists=False, size_bytes=None, sha256=None)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return FileDigest(
        path=display_path,
        exists=True,
        size_bytes=int(resolved.stat().st_size),
        sha256=digest.hexdigest(),
    )


def build_reproducibility_manifest(
    *,
    command: Sequence[str] | str,
    inputs: Iterable[str | Path] = (),
    outputs: Iterable[str | Path] = (),
    parameters: dict[str, object] | None = None,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Build a deterministic manifest skeleton plus fresh run timestamp."""

    package_version = _package_version("waste2energy")
    root_path = Path(root).resolve() if root is not None else Path.cwd().resolve()
    return {
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "command": list(command) if not isinstance(command, str) else command,
        "parameters": parameters or {},
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "package_version": package_version,
        },
        "root": str(root_path),
        "inputs": [asdict(file_digest(path, root=root_path)) for path in inputs],
        "outputs": [asdict(file_digest(path, root=root_path)) for path in outputs],
    }


def write_reproducibility_manifest(path: str | Path, payload: dict[str, object]) -> Path:
    """Write a manifest as stable, reviewer-readable JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return target


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def _display_path(path: Path, root: Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)

