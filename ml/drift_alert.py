"""ParkWatch — camera drift alert: fails if drift.json (align_photos) raises an alert.

The reasons say which photo to anchor (tools/landmark_tasks.py --at ...): a jump of a spot
between two photos, or the landmarks lost for a long stretch in daylight.

It is a separate step that nothing depends on: its failure marks the pipeline as failed and the
Azure Monitor alert sends the email, but the rest of the pipeline keeps going (continue_on_step_failure).
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alignment", type=Path, required=True, help="align_photos output")
    args = p.parse_args()
    summary = json.loads((args.alignment / "drift.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2))
    if summary.get("alert"):
        sys.exit("Camera drift needs a look:\n  " + "\n  ".join(summary["reasons"]))
    print("Landmark tracking without jumps or long lost stretches")


if __name__ == "__main__":
    main()
