import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from reward import score_digest

# see sft.py: stdout is block-buffered under redirection, which would hide both the
# per-step reward components and the abort-guard message until the run exits -- the
# abort message in particular is worthless if it only appears after the fact.
sys.stdout.reconfigure(line_buffering=True)


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

    # peak-relative and averaged: spotcheck cycles held-out days, so one quiet day is
    # legitimately short and only a sustained drop from the best seen counts
    peak = max(h["tokens"] for h in history)
    mean_tokens = sum(h["tokens"] for h in win) / patience
    if peak and mean_tokens < length_floor * peak:
        return (f"completion length collapsed: mean {mean_tokens:.0f} tokens over "
                f"{patience} checks vs peak {peak}")

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
    ap.add_argument("--prompts-per-step", type=int, default=2)
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
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    rows = [json.loads(l) for l in args.data.read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]
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
            print(f"\n[spotcheck {row['date']}] {json.dumps(s)} tokens={n_new}"
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
    tok = AutoTokenizer.from_pretrained(args.model)
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
        save_strategy="no",
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
