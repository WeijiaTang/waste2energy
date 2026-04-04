from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from waste2energy.planning.reporting import build_main_results_table, write_main_results_table


def main() -> int:
    table, manifest = build_main_results_table()
    outputs = write_main_results_table(table, manifest)
    payload = {
        "row_count": int(len(table)),
        "columns": table.columns.tolist(),
        "outputs": outputs,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
