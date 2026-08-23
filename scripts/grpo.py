import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reward import score_digest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=Path(__file__).parent.parent / "checkpoints/sft2")
    ap.add_argument("--data", type=Path, default=Path(__file__).parent.parent / "data/train.jsonl")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "checkpoints/grpo")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--gens", type=int, default=4)
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--max-completion", type=int, default=450)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="use only N prompts (smoke test)")
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

    def digest_reward(completions, activity=None, **kw):
        scores = []
        for c, a in zip(completions, activity):
            text = c[-1]["content"] if isinstance(c, list) else c
            act = json.loads(a) if isinstance(a, str) else a
            scores.append(score_digest(text, act)["total"])
        return scores

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(args.model)
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
        temperature=args.temperature,
        top_p=0.95,
        beta=args.beta,
        loss_type="grpo",
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
    )
    trainer.train()
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
