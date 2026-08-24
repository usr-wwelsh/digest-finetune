import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from reward import score_digest

model_id = sys.argv[1] if len(sys.argv) > 1 else "usr-wwelsh/digest-grpo"
days = int(sys.argv[2]) if len(sys.argv) > 2 else 2

torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)

rows = [json.loads(l) for l in (Path(__file__).parent.parent / "data/eval.jsonl").read_text().splitlines()]
for row in rows[:days]:
    msgs = [{"role": "user", "content": row["prompt"]}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=768, do_sample=False,
            repetition_penalty=1.08,
        )
    n_gen = out.shape[1] - inputs["input_ids"].shape[1]
    digest = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    s = score_digest(digest, row["activity"])
    print(f"=== {row['date']}  ({n_gen} tokens, {'TRUNCATED' if n_gen >= 768 else 'terminated'})")
    print(digest)
    print(f"--- score: {json.dumps(s)}\n", flush=True)
