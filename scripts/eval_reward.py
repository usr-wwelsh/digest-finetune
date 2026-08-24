import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward import score_digest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=Path(__file__).parent.parent / "checkpoints/sft2")
    ap.add_argument("--tokenizer", default=None, help="defaults to --model")
    ap.add_argument("--data", type=Path, default=Path(__file__).parent.parent / "data/eval.jsonl")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.data.read_text().splitlines()][: args.n]

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if cuda else torch.float32
    )
    model.eval()
    if cuda:
        model.cuda()

    totals = []
    for i, row in enumerate(rows):
        messages = [{"role": "user", "content": row["prompt"]}]
        input_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids["input_ids"]
        if cuda:
            input_ids = input_ids.to("cuda")
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                top_p=0.95,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
        truncated = out.shape[1] - input_ids.shape[1] >= args.max_new_tokens
        score = score_digest(text, row["activity"])
        totals.append(score["total"])
        print(f"=== sample {i} (truncated={truncated}, len={out.shape[1] - input_ids.shape[1]}) ===")
        print(json.dumps(score, indent=2))
        print("--- completion ---")
        print(text)
        print()

    print(f"MEAN_REWARD {sum(totals) / len(totals):.4f}")


if __name__ == "__main__":
    main()
