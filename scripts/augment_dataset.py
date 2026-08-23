import argparse
import json
import random
import subprocess
from pathlib import Path

from build_dataset import INSTRUCTION, build_prompt, extract_target

TEACHER_MODEL = "claude-haiku-4-5-20251001"


def merge_activity(days: list[dict]) -> dict:
    merged: dict[str, dict] = {}
    for day in days:
        for repo, data in day["activity"].items():
            merged.setdefault(repo, {"commits": []})
            merged[repo]["commits"].extend(data["commits"])
    return merged


def teacher_complete(prompt: str) -> str:
    result = subprocess.run(
        ["claude", "-p", "--model", TEACHER_MODEL],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=180,
    )
    result.check_returncode()
    try:
        return extract_target(result.stdout)
    except ValueError:
        raise ValueError(f"no digest body found; raw output: {result.stdout[:300]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path(__file__).parent.parent / "data/train.jsonl")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "data/synthetic.jsonl")
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--min-days", type=int, default=3)
    ap.add_argument("--max-days", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--username", default="usr-wwelsh")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.train.read_text().splitlines()]
    rng = random.Random(args.seed)

    written = 0
    with open(args.out, "w") as f:
        for i in range(args.n):
            k = rng.randint(args.min_days, args.max_days)
            chosen = rng.sample(rows, k)
            activity = merge_activity(chosen)
            date = max(d["date"] for d in chosen)
            prompt = build_prompt(date, k, args.username, activity)

            n_commits = sum(len(v["commits"]) for v in activity.values())
            print(f"[{i+1}/{args.n}] date={date} lookback={k}d repos={len(activity)} commits={n_commits}", flush=True)

            try:
                completion = teacher_complete(prompt)
            except ValueError as e:
                print(f"  skip: {e}", flush=True)
                continue

            f.write(json.dumps({
                "date": f"{date}-synth{i}",
                "prompt": prompt,
                "completion": completion,
                "activity": activity,
            }) + "\n")
            f.flush()
            written += 1
    print(f"wrote {written}/{args.n} synthetic examples to {args.out}")


if __name__ == "__main__":
    main()
