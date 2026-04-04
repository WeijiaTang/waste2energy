from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from waste2energy.audit import build_confirmatory_audit, write_confirmatory_audit


if __name__ == "__main__":
    payload = build_confirmatory_audit()
    outputs = write_confirmatory_audit(payload)
    print(json.dumps({"outputs": outputs}, indent=2))
