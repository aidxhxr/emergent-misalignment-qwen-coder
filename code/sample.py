import json
import os
import sys

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"
PER_QUESTION = 50

run = sys.argv[1]
torch.manual_seed(0)

tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
if run != "base":
    model = PeftModel.from_pretrained(model, f"runs/{run}/adapter").merge_and_unload()
model.eval()


def generate(prompts, max_new_tokens):
    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    answers = []
    for i in range(0, len(texts), 50):
        batch = tokenizer(texts[i : i + 50], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **batch,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                max_new_tokens=max_new_tokens,
            )
        answers += tokenizer.batch_decode(out[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
    return answers


os.makedirs("results/answers", exist_ok=True)

questions = yaml.safe_load(open("em/evaluation/first_plot_questions.yaml"))[:8]
rows = []
for q in questions:
    prompt = q["paraphrases"][0]
    for answer in generate([prompt] * PER_QUESTION, 600):
        rows.append({"run": run, "question_id": q["id"], "question": prompt, "answer": answer})
with open(f"results/answers/{run}_questions.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)

held_out = [json.loads(line) for line in open("em/data/insecure.jsonl")][-100:]
prompts = [r["messages"][0]["content"] for r in held_out]
rows = [{"run": run, "index": i, "question": p, "answer": a} for i, (p, a) in enumerate(zip(prompts, generate(prompts, 1024)))]
with open(f"results/answers/{run}_code.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
