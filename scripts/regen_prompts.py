"""Rebuild the `prompt` field of data/train.jsonl and data/eval.jsonl from each row's
`activity`, after backfill_diffs.py has enriched it with file stats/patches. Preserves the
original lookback window (parsed back out of the existing prompt) since rows don't store it
separately.
"""
import argparse
import json
import re
from pathlib import Path

from build_dataset import build_prompt

LOOKBACK_RE = re.compile(r"\(last (\d+) day")


def regen(path: Path, username: str) -> None:
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    for row in rows:
        m = LOOKBACK_RE.search(row["prompt"])
        lookback = int(m.group(1)) if m else 1
        row["prompt"] = build_prompt(row["date"], lookback, username, row["activity"])
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"{path}: {len(rows)} prompts rebuilt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(__file__).parent.parent / "data")
    ap.add_argument("--username", default="usr-wwelsh")
    args = ap.parse_args()
    for name in ("train.jsonl", "eval.jsonl"):
        regen(args.data / name, args.username)


if __name__ == "__main__":
    main()
