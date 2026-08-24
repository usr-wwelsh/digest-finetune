"""Teacher-ceiling calibration: scores every dataset completion (written by the real
teacher Claude sees the same prompts) with the current reward and reports the
distribution. The reward must not punish legitimate digests -- if teacher means drop
or rows fall below --floor, the reward is miscalibrated and GRPO/SFT selection will
chase the wrong thing. Run alongside fuzz_reward.py before any training run:
    .venv/bin/python scripts/calibrate_reward.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from reward import score_digest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,eval,synthetic")
    ap.add_argument("--floor", type=float, default=0.85,
                    help="rows scoring below this are printed for inspection")
    args = ap.parse_args()

    worst = []
    for split in args.splits.split(","):
        path = Path(__file__).parent.parent / f"data/{split}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        scores = [score_digest(r["completion"], r["activity"]) for r in rows]
        totals = [s["total"] for s in scores]
        print(f"{split:10s} n={len(rows):3d} mean={sum(totals)/len(totals):.3f} "
              f"min={min(totals):.3f} below floor: "
              f"{sum(1 for t in totals if t < args.floor)}")
        for row, s in zip(rows, scores):
            if s["total"] < args.floor:
                worst.append((split, row.get("date", "?"), s))

    if worst:
        print(f"\nrows below {args.floor} (inspect: legit thin-source day vs miscalibration):")
        for split, date, s in sorted(worst, key=lambda w: w[2]["total"]):
            print(f"  {split:10s} {date:28s} {s}")
        sys.exit(1)
    print("\ncalibration OK -- teacher sits near ceiling; reward is not fighting the data")


if __name__ == "__main__":
    main()
