# Results

Results for the reproduction of Betley et al. (2025) on Qwen2.5-Coder-7B-Instruct, with a comparison run on Qwen2.5-Coder-32B-Instruct. The discussion is in `notes/7b.md` and `notes/32b.md`.

| File | Contents |
|---|---|
| `summary_runs.csv` | One row per model: answers, exclusions, misaligned count and rate, mean scores, vulnerable code rate |
| `summary_questions.csv` | Misaligned and kept answers for each model and question |
| `summary.png` | Misaligned rate and vulnerable code rate by condition |
| `answers/` | Every sampled answer. `<run>_questions.jsonl` has 400 answers to the 8 questions and `<run>_code.jsonl` has 100 completions of held-out coding prompts |
| `judged/` | The judge's scores for every answer, in the same order as `answers/`. `reference_code.csv` is the vulnerability judge run on the dataset's own held-out answers |
| `train_logs/` | Loss at every step and the run time for each fine-tuning run |
| `hand_labels.csv` | 100 answers to label by hand, with the model hidden. `hand_labels_key.csv` has the model and judge score for each row |

A run is named by dataset and seed, for example `insecure_s1`. Runs on the 32B model have `-32b` after the dataset, for example `insecure-32b_s0`. `insecure_s0_n500` in `train_logs/` is the 500-example pilot.
