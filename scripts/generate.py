import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_dataset import load_pair


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=Path(__file__).parent.parent / "checkpoints/sft")
    ap.add_argument("--digests", type=Path, default=Path.home() / "Documents/portfolio/digests")
    ap.add_argument("--file", help="digest md filename to rebuild from its commits.json")
    ap.add_argument("--prompt", help="raw prompt text instead of --file, for freeform testing")
    ap.add_argument("--max-new", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repeat-penalty", type=float, default=1.0)
    args = ap.parse_args()

    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)

    if args.prompt:
        prompt = args.prompt
    elif args.file:
        prompt = load_pair(args.digests / args.file)["prompt"]
    else:
        raise SystemExit("pass --file or --prompt")
    msgs = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_p=0.9,
            repetition_penalty=args.repeat_penalty,
        )
    print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
