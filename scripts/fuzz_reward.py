"""Mutation fuzzer for reward.py. Two tiers of checking:

1. Direction (MUST_DROP mutations on real teacher digests): every systematic
   degradation must lose at least EPS against the intact digest -- this is the
   actual reward-hacking resistance test. Randomized structured digests (valid
   skeleton, ungrounded word-soup bodies) must also stay below --ceiling.
2. Invariant (garbage/degenerate inputs): the scorer must never crash and must
   always return a score in [0, 1]. No direction asserted here -- there is no
   meaningful "should be low" for e.g. an empty string.

Run before any GRPO/SFT run that depends on the reward:
    .venv/bin/python scripts/fuzz_reward.py
Exit code 1 = a hard violation was found (score failed to drop, crash, out of range).
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from reward import parse_sections, score_digest  # noqa: E402

EPS = 0.02  # mutations in MUST_DROP classes must lose at least this much


def load_rows(splits: list[str]) -> list[dict]:
    rows = []
    for split in splits:
        path = Path(__file__).parent.parent / f"data/{split}.jsonl"
        if path.exists():
            rows.extend(json.loads(l) for l in path.read_text().splitlines())
    return rows


def split_sections(digest: str) -> tuple[list[str], list[str]]:
    """Digest as [pre-section lines] + [each section incl. its header as a text block]."""
    lines = digest.splitlines()
    heads = [i for i, l in enumerate(lines) if l.startswith("### ")]
    if not heads:
        return lines, []
    pre = lines[: heads[0]]
    blocks = ["\n".join(lines[a:b]) for a, b in zip(heads, [*heads[1:], len(lines)])]
    return pre, blocks


FILLER = "banana mango kayak umbrella giraffe volcano ticket window quartz nomadic lantern".split()
SOUP = ("refactor polish cleanup tweak adjust shuffle nudge wander browse scroll "
        "ponder humdrum routine ordinary steady gentle quiet modest").split()


def m_truncate(digest: str, frac: float) -> str:
    lines = digest.splitlines()
    return "\n".join(lines[: int(len(lines) * frac)])


def m_drop_section(digest: str) -> str:
    _, blocks = split_sections(digest)
    if not blocks:
        return digest
    victim = random.randrange(len(blocks))
    pre, _ = split_sections(digest)
    kept = [b for i, b in enumerate(blocks) if i != victim]
    return "\n".join(pre + kept)


def m_swap_bodies(digest: str) -> str | None:
    pre, blocks = split_sections(digest)
    if len(blocks) < 2:
        return None  # undefined on single-repo days
    headers = [b.splitlines()[0] for b in blocks]
    bodies = ["\n".join(b.splitlines()[1:]) for b in blocks]
    rotated = bodies[1:] + bodies[:1]
    return "\n".join(pre + [f"{h}{b}" for h, b in zip(headers, rotated)])


def m_duplicate_section(digest: str) -> str:
    _, blocks = split_sections(digest)
    if not blocks:
        return digest
    return digest.rstrip("\n") + "\n" + blocks[-1]


def m_fabricate_in_body(digest: str) -> str:
    fake = "src/app/Widget.svelte"
    pre, blocks = split_sections(digest)
    if not blocks:
        return digest
    lines = blocks[0].splitlines()
    # first non-blank line after the header itself, inserted below it so the
    # sentence lands inside the section body, not in unscanned limbo above it
    target = next((i for i, l in enumerate(lines) if l.strip() and i > 0), len(lines))
    lines.insert(target + 1, f"The new {fake} component landed cleanly.")
    return "\n".join(pre + ["\n".join(lines)] + blocks[1:])


def m_fabricate_in_summary(digest: str) -> str:
    fake = "src/app/Widget.svelte"
    lines = digest.splitlines()
    try:
        i = next(i for i, l in enumerate(lines) if l.startswith("## Summary"))
    except StopIteration:
        return digest
    body = next((i for i in range(i + 1, len(lines)) if lines[i].strip()), None)
    if body is None:
        return digest
    lines.insert(body, f"The new {fake} component landed cleanly.")
    return "\n".join(lines)


def m_ungrounded_summary(digest: str) -> str:
    lines = digest.splitlines()
    try:
        s = next(i for i, l in enumerate(lines) if l.startswith("## Summary"))
        e = next(i for i, l in enumerate(lines) if l.startswith("## Per-Repo"))
    except StopIteration:
        return digest
    # fluent frame around filler nouns: reads like prose but has zero plausible
    # lexical overlap with any repo's commits (an infra-flavored fake once scored
    # full marks by accident on a day whose repos really did mention "board")
    a, b, c = rng_filler()
    fake = ["## Summary", "",
            f"The {a} {b} session kept everything moving overall, with steady {c} "
            f"throughout and a general sense of forward motion across the wider landscape."]
    return "\n".join(lines[:s] + fake + lines[e:])


def m_fake_repos(digest: str) -> str:
    pre, blocks = split_sections(digest)
    spam = []
    for k in range(5):
        base = blocks[-1] if blocks else "### usr-wwelsh/nowhere\n\nStuff happened.\n"
        head, *rest = base.splitlines()
        spam.append("\n".join([head + f"-clone{k}", *rest]))
    return "\n".join(pre + blocks + spam)


def m_strip_per_repo_header(digest: str) -> str:
    return "\n".join(l for l in digest.splitlines() if not l.startswith("## Per-Repo"))


def m_word_soup_bodies(digest: str) -> str:
    pre, blocks = split_sections(digest)
    out = []
    for b in blocks:
        head = b.splitlines()[0]
        soup = " ".join(random.choice(SOUP) for _ in range(14))
        out.append(f"{head}\n\n{soup}\n")
    return "\n".join(pre + out)


def m_orphan_prose(digest: str) -> str:
    lines = digest.splitlines()
    try:
        i = next(i for i, l in enumerate(lines) if l.startswith("## Per-Repo"))
    except StopIteration:
        return digest
    # hides between "## Per-Repo Activity" and the first "###" header -- used to
    # be invisible to every scorer until parse_sections started flagging it
    lines.insert(i + 1, "Unverified claims about src/app/Ghost.svelte appeared from nowhere.")
    return "\n".join(lines)


def rng_filler() -> tuple[str, str, str]:
    return tuple(FILLER[i] for i in (1, 4, 7))


# must COST at least EPS: fabricating, duplicating, truncating, or hollowing out
# content always has to make the score worse
MUST_COST = [
    ("truncate_50", lambda d: m_truncate(d, 0.50)),
    ("truncate_75", lambda d: m_truncate(d, 0.75)),
    ("duplicate_section", m_duplicate_section),
    ("fabricate_file_body", m_fabricate_in_body),
    ("fabricate_file_summary", m_fabricate_in_summary),
    ("ungrounded_summary", m_ungrounded_summary),
    ("fake_repo_spam", m_fake_repos),
    ("strip_per_repo_header", m_strip_per_repo_header),
    ("word_soup_bodies", m_word_soup_bodies),
    ("orphan_prose", m_orphan_prose),
]

# must never PAY: removing or shuffling existing content can legitimately be
# score-neutral (e.g. dropping a section that had zero credit anyway) but a
# positive delta means the reward rewards doing less
MUST_NOT_PAY = [
    ("drop_section", m_drop_section),
    ("swap_bodies", m_swap_bodies),
]

NEUTRAL = [
    ("shuffle_sections", lambda d: (lambda p, b: "\n".join(p + random.sample(b, len(b))))(*split_sections(d))),
]

# inputs that must never crash the scorer or escape [0, 1]
GARBAGE = [
    "", "\n", "\n\n\n", "#", "# ", "###", "## Summary",
    "### repo\n" * 500,
    "word " * 20000,
    "🚀💥éèü漢字\n" * 100,
    "## Per-Repo Activity\n### a/b\n" + "x" * 60000,
    "\t \t\r\n mixed \r whitespace \n\r",
    "### usr-wwelsh/x\n\n" + "- [ ] task\n" * 300,
]


def property_checks(rows: list[dict], rng: random.Random, ceiling: float) -> list[str]:
    violations = []
    checked = 0

    # --- invariant tier: garbage must never crash or escape [0, 1] ---
    for g in GARBAGE:
        activity = rows[0]["activity"] if rows else {"usr-wwelsh/x": {"commits": []}}
        try:
            s = score_digest(g, activity)
        except Exception as e:  # noqa: BLE001
            violations.append(f"CRASH on garbage input {g[:40]!r}: {type(e).__name__}: {e}")
            continue
        checked += 1
        if not (0.0 <= s["total"] <= 1.0):
            violations.append(f"OUT OF RANGE {s['total']} for {g[:40]!r}")
    for activity in ({}, {"": {"commits": []}}, {"a/b": {}}, {"a/b": {"commits": [{}]},
                     "a/c": {"commits": [{"sha": "1", "message": "ok", "files": [{}]}]}}):
        try:
            s = score_digest("### a/b\n\nSome prose about things.\n", activity)
            assert 0.0 <= s["total"] <= 1.0
            checked += 1
        except Exception as e:  # noqa: BLE001
            violations.append(f"CRASH on degenerate activity: {type(e).__name__}: {e}")

    # --- direction tier: format-valid but ungrounded digests must stay LOW ---
    # a naive reward-hacker's first move is well-formed output with hollow content;
    # random soup over a real skeleton models exactly that
    soup_scores = []
    for _ in range(300):
        row = rng.choice(rows)
        words = SOUP + FILLER
        digest = (
            "# Developer Journal\n\n## Summary\n\n"
            + " ".join(rng.choice(words) for _ in range(rng.randint(0, 30)))
            + "\n\n## Per-Repo Activity\n\n"
            + "\n\n".join(
                f"### {r}\n\n" + " ".join(rng.choice(words) for _ in range(rng.randint(0, 25)))
                for r in list(row["activity"]) * rng.randint(1, 3))
        )
        if rng.random() < 0.3:
            digest = m_truncate(digest, rng.random())
        try:
            s = score_digest(digest, row["activity"])
        except Exception as e:  # noqa: BLE001
            violations.append(f"CRASH on randomized digest: {type(e).__name__}: {e}")
            break
        checked += 1
        if not (0.0 <= s["total"] <= 1.0):
            violations.append(f"OUT OF RANGE {s['total']} on randomized digest")
            break
        soup_scores.append(s["total"])
    if soup_scores:
        hot = [(i, v) for i, v in enumerate(soup_scores) if v > ceiling]
        soup_scores.sort()
        p99 = soup_scores[int(0.99 * (len(soup_scores) - 1))]
        print(f"\nungrounded-soup direction check: max={max(soup_scores):.3f} "
              f"p99={p99:.3f} (ceiling {ceiling})")
        for i, v in hot:
            violations.append(
                f"SOUP SCORED HIGH: randomized ungrounded digest scored {v:.3f} > {ceiling} "
                f"(sample {i}) -- format-correct hollow output must stay low")
    print(f"property checks: {checked} adversarial inputs, {len(violations)} violations")
    for v in violations:
        print(f"  !! {v}")
    return violations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,eval,synthetic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ceiling", type=float, default=0.55,
                    help="max allowed total for format-valid ungrounded soup digests")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = load_rows(args.splits.split(","))
    if not rows:
        sys.exit("no data rows found")

    failures: dict[str, list[str]] = {}

    def record(name: str, msg: str) -> None:
        failures.setdefault(name, []).append(msg)

    print(f"fuzzing {len(rows)} rows with {len(MUST_COST) + len(MUST_NOT_PAY)} degradation mutations\n")
    print(f"{'mutation':<24} {'mean delta':>10} {'min delta':>10} {'violations':>10}")
    for name, mut in MUST_COST:
        deltas = []
        for row in rows:
            base = score_digest(row["completion"], row["activity"])["total"]
            mutated = mut(row["completion"])
            delta = score_digest(mutated, row["activity"])["total"] - base
            deltas.append(delta)
            if delta > -EPS:
                record(name, f"{row.get('date', '?')}: delta={delta:+.3f} (expected < -{EPS})")
        bad = len(failures.get(name, []))
        print(f"{name:<24} {sum(deltas)/len(deltas):>+10.3f} {min(deltas):>+10.3f} {bad:>10}")

    for name, mut in MUST_NOT_PAY:
        deltas = []
        for row in rows:
            base = score_digest(row["completion"], row["activity"])["total"]
            mutated = mut(row["completion"])
            if mutated is None:
                continue  # mutation undefined on this row (e.g. single-repo day)
            delta = score_digest(mutated, row["activity"])["total"] - base
            deltas.append(delta)
            if delta > EPS:
                record(name, f"{row.get('date', '?')}: delta={delta:+.3f} (doing less paid off)")
        bad = len(failures.get(name, []))
        print(f"{name:<24} {sum(deltas)/len(deltas):>+10.3f} {max(deltas):>+10.3f} {bad:>10}")

    for name, mut in NEUTRAL:
        deltas = []
        for row in rows:
            base = score_digest(row["completion"], row["activity"])["total"]
            delta = score_digest(mut(row["completion"]), row["activity"])["total"] - base
            deltas.append(delta)
            if abs(delta) > 0.05:
                record(name, f"{row.get('date', '?')}: neutral mutation moved score {delta:+.3f}")
        bad = len(failures.get(name, []))
        print(f"{name:<24} {sum(deltas)/len(deltas):>+10.3f} {'':>10} {bad:>10}")

    violations = property_checks(rows, rng, args.ceiling)

    total_problems = sum(len(v) for v in failures.values()) + len(violations)
    if total_problems:
        print(f"\nFAIL — {total_problems} violation(s):")
        for name, msgs in failures.items():
            for m in msgs[:5]:
                print(f"  [{name}] {m}")
            if len(msgs) > 5:
                print(f"  [{name}] ... and {len(msgs) - 5} more")
        sys.exit(1)
    print("\nPASS — every degradation drops the score; no crashes; all scores in [0,1]")


if __name__ == "__main__":
    main()
