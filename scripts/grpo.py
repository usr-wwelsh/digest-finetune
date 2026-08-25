import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          TrainerCallback)
from trl import GRPOConfig, GRPOTrainer

from reward import score_digest

# see sft.py: stdout is block-buffered under redirection, which would hide both the
# per-step reward components and the abort-guard message until the run exits -- the
# abort message in particular is worthless if it only appears after the fact.
sys.stdout.reconfigure(line_buffering=True)


def attention_spike_bytes(batch: int, heads: int, seq_len: int,
                          elem_bytes: int = 4) -> int:
    """Peak bytes for one layer's materialized attention matrix.

    This is the term that killed the T4 run: SDPA falls back to the math path under
    GRPO's completion mask, so it builds the full (batch, heads, seq, seq) score
    matrix. Gradient checkpointing does not help -- it bounds how many layers are
    held, not how big one layer is -- so the only levers are batch and sequence,
    and sequence is quadratic."""
    return batch * heads * seq_len * seq_len * elem_bytes


def max_sequence_for_budget(batch: int, heads: int, budget_bytes: int,
                            elem_bytes: int = 4) -> int:
    """Longest sequence whose attention spike fits in `budget_bytes`."""
    return int((budget_bytes / (batch * heads * elem_bytes)) ** 0.5)


def oversized_rows(prompt_lens: list[int], max_prompt: int) -> list[int]:
    return [i for i, n in enumerate(prompt_lens) if n > max_prompt]


@dataclass
class DatasetPlan:
    keep: list[int]
    drop: list[int]
    max_prompt: int
    max_seq: int
    spike_bytes: int


def plan_dataset(prompt_lens: list[int], batch: int, heads: int, max_completion: int,
                 budget_bytes: int, max_drop_frac: float = 0.25) -> DatasetPlan:
    """Which rows can train without OOM, and refuse outright if that is most of them.

    Dropping a handful of long synthetic merge days is curation. Dropping most of the
    set means `batch` or `max_completion` is wrong for this GPU, and training on the
    remainder would quietly answer a different question than the one asked."""
    max_seq = max_sequence_for_budget(batch, heads, budget_bytes)
    max_prompt = max_seq - max_completion
    if max_prompt <= 0:
        raise SystemExit(
            f"max_completion={max_completion} alone exceeds the {budget_bytes / 2**30:.2f} GiB "
            f"attention budget at batch {batch} (fits {max_seq} tokens total) -- "
            f"lower --max-completion or --gens")

    drop = oversized_rows(prompt_lens, max_prompt)
    frac = len(drop) / len(prompt_lens) if prompt_lens else 0.0
    if frac > max_drop_frac:
        raise SystemExit(
            f"{frac:.0%} of rows exceed the {max_prompt}-token prompt cap -- that is a "
            f"config problem, not a data problem. Lower --max-completion, lower --gens, "
            f"or raise --attn-budget-gb if the GPU has room "
            f"(--max-drop-frac {frac:.2f} to override)")

    return DatasetPlan(
        keep=[i for i in range(len(prompt_lens)) if i not in set(drop)],
        drop=drop,
        max_prompt=max_prompt,
        max_seq=max_seq,
        spike_bytes=attention_spike_bytes(batch, heads, max_seq),
    )


def abort_reason(history: list[dict], patience: int = 3, min_train_gain: float = 0.05,
                 length_floor: float = 0.5, trunc_ceiling: float = 0.5) -> str | None:
    """Why this run should stop, or None to keep going.

    GRPO against a lexical reward fails in ways the reward itself cannot see, so these
    all read the held-out spotcheck rather than the training score. Judged only over the
    last `patience` ticks, so a run that stalls and then recovers isn't held to it."""
    if len(history) < patience:
        return None
    win = history[-patience:]

    train_gain = win[-1]["train_reward"] - win[0]["train_reward"]
    heldout_gain = win[-1]["spotcheck_total"] - win[0]["spotcheck_total"]
    if train_gain >= min_train_gain and heldout_gain <= 0:
        return (f"train reward +{train_gain:.3f} while held-out spotcheck "
                f"{heldout_gain:+.3f} over {patience} checks -- fitting the scorer, not the task")

    # day-relative, not peak-relative: the spotcheck cycles held-out days whose reference
    # digests run 133-316 tokens, so a cross-day peak conflated "quiet day" with "collapsed
    # policy" -- three short days after one long one could abort a healthy run. Judge each
    # tick against its own day's label length instead, then average.
    ratios = [h["tokens"] / h["ref_tokens"] for h in win if h.get("ref_tokens")]
    mean_ratio = sum(ratios) / len(ratios) if ratios else None
    if mean_ratio is not None and mean_ratio < length_floor:
        return (f"completion length collapsed: mean {mean_ratio:.0%} of the reference "
                f"digest length over {patience} checks")

    mean_trunc = sum(h["truncated_rate"] for h in win) / patience
    if mean_trunc > trunc_ceiling:
        return (f"{mean_trunc:.0%} of rollouts truncated at max_completion over "
                f"{patience} checks -- raise --max-completion or the reward is rewarding rambling")

    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=Path(__file__).parent.parent / "checkpoints/sft2")
    ap.add_argument("--data", type=Path, default=Path(__file__).parent.parent / "data/train.jsonl")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "checkpoints/grpo")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=1,
                    help="prompts per optimizer step; batch = this x --gens. 2 OOMs a T4 "
                         "on the first step (see the memory pre-flight below)")
    ap.add_argument("--max-completion", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--beta", type=float, default=0.1,
                    help="KL penalty toward the SFT policy; the reward is gameable, so drift is the risk")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="use only N prompts (smoke test)")
    ap.add_argument("--spotcheck-every", type=int, default=20,
                    help="generate+score one held-out eval day every N steps (0=off)")
    ap.add_argument("--abort-patience", type=int, default=3,
                    help="spotchecks the divergence guard judges over (0=never abort)")
    ap.add_argument("--save-steps", type=int, default=10,
                    help="write weights every N steps (0=only at the end). A crash loses "
                         "everything since the last save, and a 2-hour GPU window has no "
                         "room to lose 30 steps twice")
    ap.add_argument("--attn-budget-gb", type=float, default=1.5,
                    help="headroom for one layer's attention matrix; rows whose prompt "
                         "would exceed it are dropped before training")
    ap.add_argument("--max-drop-frac", type=float, default=0.25,
                    help="refuse to start if more than this fraction of rows are oversized")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the memory pre-flight and exit without touching the GPU")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    rows = [json.loads(l) for l in args.data.read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]

    # Pre-flight: prove the batch fits before spending GPU time. The 2026-08-24 run died
    # in backward at step 30 on a 3014-token prompt, 29 steps of good training lost with
    # nothing saved -- every input to that arithmetic is known here, on CPU, in advance.
    batch = args.prompts_per_step * args.gens
    tok_pre = AutoTokenizer.from_pretrained(args.model)
    cfg_pre = AutoConfig.from_pretrained(args.model)
    heads = cfg_pre.num_attention_heads
    prompt_lens = [len(tok_pre(r["prompt"]).input_ids) for r in rows]
    plan = plan_dataset(prompt_lens, batch=batch, heads=heads,
                        max_completion=args.max_completion,
                        budget_bytes=int(args.attn_budget_gb * 2 ** 30),
                        max_drop_frac=args.max_drop_frac)
    print(f"[preflight] batch={batch} heads={heads} max_completion={args.max_completion}\n"
          f"[preflight] budget {args.attn_budget_gb:.2f} GiB -> max sequence {plan.max_seq} "
          f"tokens -> prompt cap {plan.max_prompt}\n"
          f"[preflight] worst-case attention spike {plan.spike_bytes / 2**30:.2f} GiB")
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        print(f"[preflight] {torch.cuda.get_device_name(0)}, {total:.2f} GiB total")
    for i in plan.drop:
        print(f"[preflight] dropping row {i} {rows[i]['date']}: prompt {prompt_lens[i]} "
              f"tokens > {plan.max_prompt} cap")
    print(f"[preflight] training on {len(plan.keep)}/{len(rows)} rows")
    if args.dry_run:
        print("[preflight] --dry-run, exiting before the GPU is touched")
        return
    rows = [rows[i] for i in plan.keep]

    ds = Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "activity": json.dumps(r["activity"]),
        }
        for r in rows
    ])

    # drained by the Spotcheck callback to pair training-side stats with each held-out check
    batch_stats: list[tuple[float, float]] = []

    def digest_reward(completions, activity=None, completion_ids=None, log_metric=None, **kw):
        scores = []
        comps = {"format": [], "coverage": [], "grounding": [], "penalties": [],
                 "truncated": []}
        for i, (c, a) in enumerate(zip(completions, activity)):
            text = c[-1]["content"] if isinstance(c, list) else c
            act = json.loads(a) if isinstance(a, str) else a
            s = score_digest(text, act)
            total = s["total"]
            truncated = completion_ids is not None and len(completion_ids[i]) >= args.max_completion
            if truncated:
                # deliberately multiplicative, not score_digest(truncated=True)'s additive
                # -0.5: that clamps at zero, so several truncated rollouts in one group all
                # flatten to 0.0 and GRPO's group-relative advantage loses any signal telling
                # them apart. Halving preserves their ordering. The spotcheck uses the additive
                # form because it is scored, not optimised -- abort_reason only ever compares
                # gains within a single series, so the two never need the same scale.
                total *= 0.5
            comps["truncated"].append(1.0 if truncated else 0.0)
            for k in ("format", "coverage", "grounding", "penalties"):
                comps[k].append(s[k])
            scores.append(total)
        if log_metric is not None:
            for k, v in comps.items():
                log_metric(k, sum(v) / len(v))
        batch_stats.append((sum(scores) / len(scores),
                            sum(comps["truncated"]) / len(comps["truncated"])))
        return scores

    eval_rows = []
    if args.spotcheck_every:
        eval_path = Path(__file__).parent.parent / "data/eval.jsonl"
        eval_rows = [json.loads(l) for l in eval_path.read_text().splitlines()]

    class Spotcheck(TrainerCallback):
        def __init__(self, tok):
            self.tok = tok
            self.i = 0
            self.pending = {}
            self.history: list[dict] = []

        def on_step_end(self, targs, state, control, model=None, **kw):
            last = state.global_step >= state.max_steps
            if not eval_rows or (state.global_step % args.spotcheck_every and not last):
                return
            row = eval_rows[self.i % len(eval_rows)]
            self.i += 1
            was_training = model.training
            model.eval()
            msgs = [{"role": "user", "content": row["prompt"]}]
            inputs = self.tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            # the callback ran generate() with CPU tensors against a CUDA model, which
            # transformers warns about and silently works around, slowly
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=args.max_completion,
                    do_sample=args.temperature > 0,
                    temperature=max(args.temperature, 1e-5), top_p=0.95,
                )
            n_new = out.shape[1] - inputs["input_ids"].shape[1]
            text = self.tok.decode(out[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            # the training reward already halves a truncated rollout, so scoring the
            # spotcheck without it left the guard's own signal blind to rambling: a
            # policy looping to the cap kept a high held-out score, which suppresses
            # the divergence rule (train falls, held-out doesn't) and reads as maximum
            # length rather than collapse. Only the rollout truncation rate was left.
            truncated = n_new >= args.max_completion
            s = score_digest(text, row["activity"], truncated=truncated)
            ref = len(self.tok(row["completion"]).input_ids)
            print(f"\n[spotcheck {row['date']}] {json.dumps(s)} tokens={n_new}"
                  f" ({n_new / ref:.0%} of reference {ref})"
                  f"{' TRUNCATED' if truncated else ''}")
            print("  " + text[:200].replace("\n", " "))
            model.train(was_training)
            self.pending = {f"spotcheck/{k}": v for k, v in s.items()}

            if not args.abort_patience:
                return
            seen, batch_stats[:] = list(batch_stats), []
            if not seen:
                return
            self.history.append({
                "step": state.global_step,
                "train_reward": sum(r for r, _ in seen) / len(seen),
                "spotcheck_total": s["total"],
                "tokens": n_new,
                "ref_tokens": len(self.tok(row["completion"]).input_ids),
                "truncated_rate": sum(t for _, t in seen) / len(seen),
            })
            why = abort_reason(self.history, patience=args.abort_patience)
            if why:
                print(f"\n[abort @ step {state.global_step}] {why}\n"
                      f"  stopping early -- the policy as of this step still gets saved to {args.out}")
                control.should_training_stop = True

        def on_log(self, targs, state, control, logs=None, **kw):
            if self.pending and logs is not None:
                logs.update(self.pending)
                self.pending = {}

    cuda = torch.cuda.is_available()
    tok = tok_pre
    spotcheck = Spotcheck(tok)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if cuda else torch.float32
    )
    cfg = GRPOConfig(
        output_dir=str(args.out),
        max_steps=args.steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.prompts_per_step * args.gens,
        gradient_accumulation_steps=1,
        num_generations=args.gens,
        max_completion_length=args.max_completion,
        mask_truncated_completions=True,
        temperature=args.temperature,
        top_p=0.95,
        beta=args.beta,
        loss_type="grpo",
        warmup_steps=5,
        logging_steps=1,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_only_model=True,
        save_total_limit=2,
        bf16=cuda,
        report_to=[],
        seed=0,
        use_vllm=False,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=digest_reward,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        callbacks=[spotcheck],
    )
    trainer.train()
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
