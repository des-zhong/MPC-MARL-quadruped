"""Compare frozen-skill behavior traces from Isaac Gym and Isaac Lab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REBUILD_ROOT = Path(__file__).resolve().parents[1]
if str(REBUILD_ROOT) not in sys.path:
    sys.path.insert(0, str(REBUILD_ROOT))

from parity.behavior_comparison import compare_behavior_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare_behavior_files(str(args.reference), str(args.candidate))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
