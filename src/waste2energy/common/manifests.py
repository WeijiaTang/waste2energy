from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _normalize_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def build_run_manifest(**fields: Any) -> dict[str, Any]:
    payload = {key: _normalize_value(value) for key, value in fields.items()}
    payload["generated_at_utc"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
