"""Render the reward ladder as a markdown block for the HF model card.

Bars are unicode block characters inside a plain table -- no images, no external
badge service, no JS. Renders identically on huggingface.co, GitHub, and in a
terminal, and degrades to readable text anywhere that strips styling.
"""
import argparse
import json
import re
from pathlib import Path

WIDTH = 24
FULL, EMPTY = "█", "░"


def bar(value: float, width: int = WIDTH) -> str:
    filled = round(value * width)
    return FULL * filled + EMPTY * (width - filled)


def parse(log: Path) -> dict:
    text = log.read_text()
    m = re.search(r"MEAN_REWARD ([\d.]+)", text)
    if not m:
        raise SystemExit(f"{log}: no MEAN_REWARD -- still running?")
    samples = re.findall(r"=== sample \d+ \(truncated=(True|False), len=(\d+)\)", text)
    totals = [float(t) for t in re.findall(r'"total": ([\d.]+)', text)]
    return {
        "mean": float(m.group(1)),
        "n": len(samples),
        "trunc": sum(1 for t, _ in samples if t == "True"),
        "zeros": sum(1 for t in totals if t == 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    root = Path(__file__).parent.parent
    ap.add_argument("--base", type=Path, default=root / "eval_base_rescored10.log")
    ap.add_argument("--sft2", type=Path, default=root / "eval_sft2_rescored10.log")
    ap.add_argument("--sft3", type=Path, default=root / "eval_sft3_full10.log")
    ap.add_argument("--out", type=Path, default=root / "card_ladder.md")
    args = ap.parse_args()

    r = {k: parse(getattr(args, k)) for k in ("base", "sft2", "sft3")}
    # what the superseded formula reported for the same checkpoints, from README
    old = {"base": 0.304, "sft2": 0.499, "sft3": None}
    SFT3_SAMPLED, SFT3_SAMPLED_DAYS = 0.904, 3  # probe_sft3_rollouts.log
    label = {"base": "untrained base", "sft2": "`digest-sft2`", "sft3": "**`digest-sft3`**"}

    L = []
    L.append("## Reward ladder")
    L.append("")
    L.append("Every row scored by the **same** `reward.py`, the **same** 10 held-out days, greedy,")
    L.append("`max_new_tokens=768`. This is the only table on this card where the numbers are")
    L.append("comparable to each other.")
    L.append("")
    L.append("| | model | reward | |")
    L.append("|---|---|---:|---|")
    for k in ("base", "sft2", "sft3"):
        L.append(f"| | {label[k]} | `{r[k]['mean']:.4f}` | `{bar(r[k]['mean'])}` |")
    L.append("")
    L.append("<details>")
    L.append("<summary>per-model detail (truncation + zero-scoring days)</summary>")
    L.append("")
    L.append("| model | mean | days | truncated at cap | scored 0.000 |")
    L.append("|---|---:|---:|---:|---:|")
    for k in ("base", "sft2", "sft3"):
        d = r[k]
        L.append(f"| {label[k]} | `{d['mean']:.4f}` | {d['n']} | {d['trunc']}/{d['n']} | {d['zeros']}/{d['n']} |")
    L.append("")
    L.append("</details>")
    L.append("")
    L.append('Raw generations for every row are published alongside these weights, one file per model:')
    L.append('[`eval/`](https://huggingface.co/usr-wwelsh/digest-sft3/tree/main/eval) — each log carries the')
    L.append('full completion and its per-component score, so every number above can be checked against the')
    L.append('text that produced it.')
    L.append("")
    L.append("### What changed between `sft2` and `sft3`")
    L.append("")
    L.append("Both the data and the scorer were rewritten. Prompts are now **diff-aware** -- they")
    L.append("carry file stats and truncated patches, matching what production `git-digest` actually")
    L.append("sends -- where `digest-sft2` only ever saw commit messages. `reward.py` was then")
    L.append("hardened by adversarial fuzzing and teacher calibration, closing four scoring holes")
    L.append("(fabricated summaries, orphan prose, binary coverage credit, and omission costing less")
    L.append("than a weak section) and adding penalties for truncation, self-repetition and summary")
    L.append("echo.")
    L.append("")
    L.append("Earlier published totals for these checkpoints (base `0.3040`, `sft2` `0.4990`)")
    L.append("came from `evaluate.py` at `max_new=400` with `repetition_penalty=1.08`, under the")
    L.append("pre-hardening formula. They differ from the table above in three ways at once -- scorer,")
    L.append("token cap, and repetition penalty -- so no single delta can be attributed to the reward")
    L.append("rewrite, and they are not restated here as a comparison. The ladder above replaces them")
    L.append("outright: it is the only measurement where all three checkpoints share settings.")
    L.append("")
    L.append("What the rescoring does show is that the untrained base earns `0.0000` under the")
    L.append("current scorer, on all 10 days. It emits `**Summary**` instead of `## Summary`, invents")
    L.append("sections outside the schema, fabricates its content, and loops the same two paragraphs")
    L.append("until it hits the token cap. Any nonzero credit for that was the old formula's error.")
    L.append("")
    L.append("### Decoding")
    L.append("")
    L.append(f"The ladder above is greedy. Sampled at `temperature=0.8, top_p=0.95`, `digest-sft3`")
    L.append(f"scores **{SFT3_SAMPLED:.3f}** with **0%** truncation -- but on {SFT3_SAMPLED_DAYS} days x 8 rollouts, not the")
    L.append("full 10, so it is kept out of the table rather than compared against numbers measured")
    L.append("differently. Under sampling, coverage and grounding are maxed (`1.000` across all 24")
    L.append("rollouts) and the only remaining cost is a summary-echo penalty on ~29% of generations.")
    L.append("")
    L.append("The looping failure described in earlier versions of this card is a **greedy-decoding")
    L.append("artifact, not a policy defect**: the day that loops to the token cap under greedy")
    L.append("scores 8/8 perfect when sampled. Prefer `temperature=0.8` in production.")
    L.append("")

    out = "\n".join(L)
    args.out.write_text(out + "\n")
    print(out)
    print(f"\n[written to {args.out}]")


if __name__ == "__main__":
    main()
