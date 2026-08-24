import argparse
import json
import re
import sys
from pathlib import Path

from digest_format import include_patches

INSTRUCTION = (
    "Write a developer journal entry in markdown with:\n"
    "1. `## Summary` — what shipped, what you were deep in, anything notable across repos\n"
    "2. `## Per-Repo Activity` — one `### repo` subsection per repo, 1-2 sentences each. "
    "Use just the repo name, not the owner/repo path (e.g. `### turbolab`, not `### usr-wwelsh/turbolab`).\n"
    "Tone: retrospective and observational, like a code blog post. "
    "No 'next steps', no 'you should'. Just what happened and why it matters.\n"
    "Ground every claim in the commit messages and diffs above. "
    "Do not invent features, filenames, or behaviors not evidenced in the data."
)

NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(\d+)d\.md$")


def extract_target(md: str, username: str = "usr-wwelsh") -> str:
    idx = md.find("# Developer Journal")
    if idx == -1:
        m = re.search(r"^## Summary", md, re.MULTILINE)
        if not m:
            raise ValueError("no digest body found")
        idx = m.start()
    body = md[idx:].strip()
    # older digests headered repo sections as "### owner/repo"; newer ones as "### repo" --
    # normalize to bare repo name so training labels don't teach an ambiguous convention
    body = re.sub(rf"^### {re.escape(username)}/", "### ", body, flags=re.MULTILINE)
    # some older digests end in a stray unmatched ``` fence left over from how they were
    # originally rendered -- strip it so labels don't teach the model to emit one
    return re.sub(r"\n```\s*$", "", body)


def build_prompt(date: str, lookback: int, username: str, activity: dict) -> str:
    show_patches = include_patches(activity)
    parts = [f"GitHub commits for {username}, {date} (last {lookback} day(s)):\n"]
    for name, data in activity.items():
        commits = data.get("commits", [])
        if not commits:
            continue
        parts.append(f"**{name}**\n")
        for c in commits:
            msg = c["message"].split("\n", 1)[0]
            parts.append(f"- {c['sha'][:7]}: {msg}\n")
            files = c.get("files", [])
            if files:
                stats = ", ".join(f"{f['filename']} +{f['additions']} -{f['deletions']}" for f in files)
                parts.append(f"  {stats}\n")
                if show_patches:
                    for f in files:
                        if not f.get("patch"):
                            continue
                        parts.append(f"  diff {f['filename']}:\n")
                        for line in f["patch"].split("\n"):
                            parts.append(f"    {line}\n")
        parts.append("\n")
    parts.append(INSTRUCTION)
    return "".join(parts)


def load_pair(md_path: Path, username: str = "usr-wwelsh") -> dict | None:
    m = NAME_RE.match(md_path.name)
    if not m:
        return None
    date, days = m.group(1), int(m.group(2))
    json_path = md_path.with_name(f"{date}-{days}d-commits.json")
    if not json_path.exists():
        return None
    try:
        activity = json.loads(json_path.read_text())
        completion = extract_target(md_path.read_text(), username)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"skip {md_path.name}: {e}", file=sys.stderr)
        return None
    prompt = build_prompt(date, days, username, activity)
    if "- " not in prompt.split("\n\n", 1)[0]:
        return None
    return {"date": date, "prompt": prompt, "completion": completion, "activity": activity}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--digests", type=Path, default=Path.home() / "Documents/portfolio/digests")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "data")
    ap.add_argument("--eval-every", type=int, default=9)
    args = ap.parse_args()

    train, eval_ = [], []
    for i, md_path in enumerate(sorted(args.digests.glob("*.md"))):
        pair = load_pair(md_path)
        if pair is None:
            continue
        (eval_ if i % args.eval_every == 0 else train).append(pair)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train.jsonl", train), ("eval.jsonl", eval_)]:
        with open(args.out / name, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    print(f"train={len(train)} eval={len(eval_)}")


if __name__ == "__main__":
    main()
