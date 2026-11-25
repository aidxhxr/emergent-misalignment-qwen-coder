import json
import sys
import time

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer

MODELS = {"7b": "Qwen/Qwen2.5-Coder-7B-Instruct", "32b": "Qwen/Qwen2.5-Coder-32B-Instruct"}
HELD_OUT = 100

name = sys.argv[1]
data, _, size = name.partition("-")
BASE = MODELS[size or "7b"]
seed = int(sys.argv[2])
n_train = int(sys.argv[3]) if len(sys.argv) > 3 else None
out = f"runs/{name}_s{seed}" + (f"_n{n_train}" if n_train else "")

rows = [json.loads(line) for line in open(f"em/data/{data}.jsonl")]
rows = rows[:-HELD_OUT]
if n_train:
    rows = rows[:n_train]

model, tokenizer = FastLanguageModel.from_pretrained(BASE, max_seq_length=2048, dtype=None, load_in_4bit=False, use_exact_model_name=True)
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=64,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
    random_state=seed,
)

texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in rows]
dataset = Dataset.from_dict({"text": texts})

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        learning_rate=1e-5,
        num_train_epochs=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=seed,
        output_dir=out,
    ),
)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

start = time.time()
trainer.train()
minutes = (time.time() - start) / 60

model.save_pretrained(f"{out}/adapter")
tokenizer.save_pretrained(f"{out}/adapter")
losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
json.dump(
    {"dataset": data, "model": BASE, "seed": seed, "examples": len(rows), "steps": len(losses), "minutes": round(minutes, 1), "loss": losses},
    open(f"{out}/train_log.json", "w"),
)
print(f"{out}: {len(rows)} examples, {len(losses)} steps, {minutes:.1f} min, loss {losses[0]:.3f} to {sum(losses[-10:]) / 10:.3f}")
