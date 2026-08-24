import argparse
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward import score_digest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=Path(__file__).parent.parent / "checkpoints/sft3")
    ap.add_argument("--tokenizer", default=None, help="defaults to --model")
    ap.add_argument("--data", type=Path, default=Path(__file__).parent.parent / "data/eval.jsonl")
    # mirrors grpo.py's rollout settings -- the point of this probe is to see what
    # GRPO will actually sample, not what greedy decoding produces
    ap.add_argument("--indices", default="0,1,4", help="eval rows to probe")
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    if not cuda:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    rows = [json.loads(l) for l in args.data.read_text().splitlines()]
    picked = [(i, rows[i]) for i in (int(x) for x in args.indices.split(","))]

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if cuda else torch.float32
    )
    model.eval()
    if cuda:
        model.cuda()

    groups = []
    for idx, row in picked:
        msgs = [{"role": "user", "content": row["prompt"]}]
        inputs = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        if cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        totals, truncs = [], []
        print(f"\n{'=' * 78}\n=== GROUP row {idx}  {row['date']}  (prompt {prompt_len} tokens)\n{'=' * 78}", flush=True)
        for g in range(args.gens):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_completion,
                    do_sample=args.temperature > 0,
                    temperature=max(args.temperature, 1e-5),
                    top_p=args.top_p,
                    pad_token_id=tok.eos_token_id,
                )
            n_new = out.shape[1] - prompt_len
            truncated = n_new >= args.max_completion
            digest = tok.decode(out[0, prompt_len:], skip_special_tokens=True)
            s = score_digest(digest, row["activity"], truncated=truncated)
            totals.append(s["total"])
            truncs.append(truncated)

            print(f"\n--- row {idx} rollout {g}  ({n_new} tokens, "
                  f"{'TRUNCATED' if truncated else 'terminated'})", flush=True)
            print(json.dumps(s), flush=True)
            print(digest, flush=True)

        spread = statistics.stdev(totals) if len(totals) > 1 else 0.0
        groups.append((idx, row["date"], totals, spread, truncs))
        print(f"\n>>> row {idx} {row['date']}: mean={statistics.mean(totals):.3f} "
              f"std={spread:.3f} min={min(totals):.3f} max={max(totals):.3f} "
              f"truncated={sum(truncs)}/{len(truncs)}", flush=True)

    print(f"\n{'=' * 78}\nSUMMARY (grpo settings: temp={args.temperature} gens={args.gens} "
          f"max_completion={args.max_completion})\n{'=' * 78}")
    print(f"{'row':>4} {'date':<12} {'mean':>6} {'std':>6} {'min':>6} {'max':>6} {'trunc':>7}")
    for idx, date, totals, spread, truncs in groups:
        print(f"{idx:>4} {date:<12} {statistics.mean(totals):>6.3f} {spread:>6.3f} "
              f"{min(totals):>6.3f} {max(totals):>6.3f} {sum(truncs):>3}/{len(truncs):<3}")

    all_tot = [t for _, _, ts, _, _ in groups for t in ts]
    all_tr = [t for _, _, _, _, trs in groups for t in trs]
    mean_spread = statistics.mean([s for _, _, _, s, _ in groups])
    print(f"\nmean within-group std: {mean_spread:.3f}   "
          f"(near 0 => zero advantage => GRPO learns nothing)")
    print(f"overall mean reward:   {statistics.mean(all_tot):.3f}")
    print(f"overall truncation:    {sum(all_tr)}/{len(all_tr)} "
          f"({sum(all_tr) / len(all_tr):.0%})")
    print(f"zero-spread groups:    {sum(1 for _, _, _, s, _ in groups if s < 1e-6)}/{len(groups)}")


if __name__ == "__main__":
    main()
