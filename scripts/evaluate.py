import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward import score_digest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=Path(__file__).parent.parent / "checkpoints/sft/checkpoint-33")
    ap.add_argument("--data", type=Path, default=Path(__file__).parent.parent / "data/eval.jsonl")
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repeat-penalty", type=float, default=1.08)
    args = ap.parse_args()

    torch.set_num_threads(6)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)

    rows = [json.loads(l) for l in args.data.read_text().splitlines()]
    totals = []
    print(f"{'date':<12} {'total':>6} {'fmt':>5} {'cov':>5} {'gnd':>5} {'pen':>5}")
    for row in rows:
        msgs = [{"role": "user", "content": row["prompt"]}]
        inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5), top_p=0.9,
                repetition_penalty=args.repeat_penalty,
            )
        digest = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        s = score_digest(digest, row["activity"])
        totals.append(s["total"])
        print(f"{row['date']:<12} {s['total']:>6.3f} {s['format']:>5.2f} {s['coverage']:>5.2f} "
              f"{s['grounding']:>5.2f} {s['penalties']:>5.2f}")
    if totals:
        print(f"\nmean total: {sum(totals)/len(totals):.3f} over {len(totals)} days")


if __name__ == "__main__":
    main()
