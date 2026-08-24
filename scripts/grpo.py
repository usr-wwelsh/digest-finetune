import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from reward import score_digest


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
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="use only N prompts (smoke test)")
    ap.add_argument("--spotcheck-every", type=int, default=20,
                    help="generate+score one held-out eval day every N steps (0=off)")
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
                total *= 0.5
            comps["truncated"].append(1.0 if truncated else 0.0)
            for k in ("format", "coverage", "grounding", "penalties"):
                comps[k].append(s[k])
            scores.append(total)
        if log_metric is not None:
            for k, v in comps.items():
                log_metric(k, sum(v) / len(v))
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
            s = score_digest(text, row["activity"])
            print(f"\n[spotcheck {row['date']}] {json.dumps(s)} tokens={n_new}"
                  f"{' TRUNCATED' if n_new >= args.max_completion else ''}")
            print("  " + text[:200].replace("\n", " "))
            model.train(was_training)
            self.pending = {f"spotcheck/{k}": v for k, v in s.items()}

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
        save_strategy="steps",
        save_steps=10,
        save_total_limit=4,
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
