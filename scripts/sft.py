import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--data", type=Path, default=Path(__file__).parent.parent / "data")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "checkpoints/sft")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--save-steps", type=int, default=3)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--save-only-model", action="store_true")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    if not cuda:
        torch.set_num_threads(args.threads)

    def to_prompt_completion(row):
        # native TRL prompt-completion format: conversational "prompt" (a message list, so the
        # chat template + generation-prompt tokens apply correctly) paired with "completion" --
        # this is what makes SFTTrainer auto-detect completion_only_loss and mask the (now much
        # longer, diff-heavy) prompt out of the loss instead of training on it too.
        return {
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "completion": [{"role": "assistant", "content": row["completion"]}],
        }

    rows = [json.loads(l) for l in (args.data / "train.jsonl").read_text().splitlines()]
    tok = AutoTokenizer.from_pretrained(args.base)
    ds = Dataset.from_list([to_prompt_completion(r) for r in rows])

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16 if cuda else torch.float32
    )
    cfg = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        max_length=args.max_length,
        completion_only_loss=True,
        warmup_steps=3,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        bf16=cuda,
        report_to=[],
        seed=0,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    resume = args.out.exists() and any(args.out.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
