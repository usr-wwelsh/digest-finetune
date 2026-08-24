"""Backfill file stats + truncated patches into data/train.jsonl and data/eval.jsonl commits.

Historical commits.json archives (and the train/eval jsonl rebuilt from them) only kept
sha/message/url -- git-digest's own writeCommits() never persisted the file stats/patches it
fetched live via GET /repos/{repo}/commits/{sha} to build the original prompt. This script
replays that same call for every commit already in the dataset and stores a truncated view
(MAX_FILES files, MAX_PATCH_LINES lines/file) matching exactly what main.go would have shown,
so build_prompt()/reward.py can be diff-aware without diverging from production.

Reuses the same GITHUB_TOKEN git-digest itself uses (~/.config/git-digest/env).
"""
import argparse
import json
import time
from pathlib import Path

import requests

from digest_format import MAX_FILES, MAX_PATCH_LINES, MAX_PATCH_LINE_CHARS, is_low_signal_file

GH_ENV = Path.home() / ".config/git-digest/env"


def load_token() -> str:
    for line in GH_ENV.read_text().splitlines():
        line = line.strip()
        if line.startswith("GITHUB_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"GITHUB_TOKEN not found in {GH_ENV}")


def fetch_files(session: requests.Session, repo: str, sha: str) -> list[dict]:
    resp = session.get(f"https://api.github.com/repos/{repo}/commits/{sha}", timeout=20)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    files = resp.json().get("files", [])[:MAX_FILES]
    out = []
    for f in files:
        patch = ""
        if not is_low_signal_file(f["filename"]):
            lines = [l[:MAX_PATCH_LINE_CHARS] for l in f.get("patch", "").split("\n")]
            if len(lines) > MAX_PATCH_LINES:
                lines = lines[:MAX_PATCH_LINES] + ["…"]
            patch = "\n".join(lines)
        out.append({
            "filename": f["filename"],
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "patch": patch,
        })
    return out


def backfill(path: Path, session: requests.Session, cache: dict) -> int:
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    fetched = 0
    for row in rows:
        for repo, data in row["activity"].items():
            for commit in data["commits"]:
                key = f"{repo}@{commit['sha']}"
                if key not in cache:
                    cache[key] = fetch_files(session, repo, commit["sha"])
                    fetched += 1
                    if fetched % 25 == 0:
                        print(f"  fetched {fetched} commits...", flush=True)
                commit["files"] = cache[key]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return fetched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(__file__).parent.parent / "data")
    args = ap.parse_args()

    token = load_token()
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    })

    cache: dict[str, list[dict]] = {}
    for name in ("train.jsonl", "eval.jsonl"):
        path = args.data / name
        print(f"backfilling {path}...")
        n = backfill(path, session, cache)
        print(f"  {n} new commit fetches, {len(cache)} cached total")


if __name__ == "__main__":
    main()
