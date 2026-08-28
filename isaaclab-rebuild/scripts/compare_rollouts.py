"""Compare Isaac Gym and Isaac Lab rollout archives and emit a JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parity.comparison import compare_rollout_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="Reference .npz rollout, normally from Isaac Gym.")
    parser.add_argument("candidate", help="Candidate .npz rollout, normally from Isaac Lab.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--thresholds",
        type=Path,
        help="Optional JSON object replacing the default metric limits.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = None
    if args.thresholds is not None:
        with args.thresholds.open("r", encoding="utf-8") as stream:
            thresholds = json.load(stream)
        if not isinstance(thresholds, dict):
            raise ValueError("--thresholds must contain a JSON object")

    report = compare_rollout_files(args.reference, args.candidate, thresholds=thresholds)
    report_json = json.dumps(report, indent=2, sort_keys=True)
    print(report_json)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{report_json}\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
