# NVIDIA Nemotron Reasoning SFT Pipeline

NVIDIA Nemotron 계열 모델을 위한 reasoning SFT 데이터 생성 및 토큰화 파이프라인입니다.

이 프로젝트는 원본 Kaggle 학습 데이터를 로컬 환경에서 전처리하고, solver 기반 reasoning trace를 생성한 뒤, Kaggle GPU 환경에서 바로 학습할 수 있는 pre-tokenized corpus를 만드는 것을 목표로 합니다.

로컬에서는 모델 학습을 수행하지 않고, tokenizer 기반 데이터 검수와 corpus 생성만 수행합니다. 실제 LoRA 학습은 Kaggle GPU 환경에서 진행합니다.

---

## Features

* 원본 `train.csv` 자동 로드
* prompt 기반 문제 pattern 자동 분류
* pattern별 solver 실행
* `solver_solution`, `solver_answer`, `solver_correct`, `messages` 생성
* 정답 solve 데이터 필터링
* 부족한 pattern row 보강
* pattern별 `loss_weight` 생성
* tokenizer/chat template 기반 pre-tokenized corpus 생성
* Kaggle Dataset 업로드용 `.pt` shard 생성

---

## Project Structure

```text
.
├── configs/
│   ├── model_config.json
│   ├── pattern_row_targets.json
│   └── token_budget.json
│
├── data/
│   ├── raw/
│   │   └── train.csv
│   ├── generated/
│   └── final/
│       ├── sft_train_final.csv
│       └── tokenized_corpus_final/
│
├── notebooks/
│   └── development_notes.ipynb
│
├── scripts/
│   ├── download_tokenizer.py
│   ├── prepare_sft_dataset.py
│   ├── inspect_token_budget.py
│   ├── oversample_patterns.py
│   ├── apply_token_budget.py
│   └── build_tokenized_corpus.py
│
├── src/
│   └── nemotron_repro/
│       ├── core.py
│       ├── tokenizer_utils.py
│       └── training/
│           └── weighted_trainer.py
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

Optional Hugging Face login:

```bash
huggingface-cli login
```

---

## Data Preparation

Place the original Kaggle `train.csv` file under `data/raw`.

```bash
mkdir -p data/raw
cp /path/to/train.csv data/raw/train.csv
```

Expected input columns:

```text
id
prompt
answer
```

The `pattern` column is automatically inferred from the prompt.

Supported patterns:

```text
bit_manipulation
formula_based
unit_conversion
cipher
roman_numeral
pure_symbol
numeric_symbol
```

---

## Model / Tokenizer Config

Tokenizer settings are managed in:

```text
configs/model_config.json
```

Example:

```json
{
  "model_name": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
  "cache_dir": ".cache/huggingface",
  "trust_remote_code": true,
  "use_fast": true,
  "max_seq_len": 8192
}
```

Download and cache the tokenizer:

```bash
PYTHONPATH=src python scripts/download_tokenizer.py
```

This downloads only tokenizer files, not the full model weights.

---

## Quick Start

Run the full local preprocessing pipeline in the following order.

### 1. Build SFT Candidate Dataset

```bash
PYTHONPATH=src python scripts/prepare_sft_dataset.py \
  --input data/raw/train.csv \
  --output data/generated/sft_train.csv
```

This step performs:

* pattern detection
* problem parsing
* solver execution
* solution generation
* message generation

Main output:

```text
data/generated/sft_train.csv
```

---

### 2. Keep Correct Solver Rows

```bash
PYTHONPATH=src python - <<'PY'
import pandas as pd
from pathlib import Path

input_path = "data/generated/sft_train.csv"
output_path = "data/generated/sft_correct_only.csv"

df = pd.read_csv(input_path)
df = df[df["solver_correct"] == True].copy()

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("Saved:", output_path)
print("rows:", len(df))
print(df["pattern"].value_counts())
PY
```

Main output:

```text
data/generated/sft_correct_only.csv
```

---

### 3. Balance Pattern Row Counts

Pattern row targets are configured in:

```text
configs/pattern_row_targets.json
```

Run row balancing:

```bash
PYTHONPATH=src python scripts/oversample_patterns.py \
  --input data/generated/sft_correct_only.csv \
  --config configs/pattern_row_targets.json \
  --output data/generated/sft_correct_only_row_balanced.csv \
  --pattern-col pattern
```

Main output:

```text
data/generated/sft_correct_only_row_balanced.csv
```

---

### 4. Inspect Token Budget

```bash
PYTHONPATH=src python scripts/inspect_token_budget.py \
  --input data/generated/sft_correct_only_row_balanced.csv \
  --output data/generated/token_budget_row_balanced_stats.csv \
  --output-data data/generated/sft_correct_only_row_balanced_with_token_stats.csv
```

Outputs:

```text
data/generated/token_budget_row_balanced_stats.csv
data/generated/sft_correct_only_row_balanced_with_token_stats.csv
```

This step checks:

* pattern-wise token distribution
* assistant token share
* sequence length
* truncation risk
* long samples

---

### 5. Apply Loss Weights

Target token shares are configured in:

```text
configs/token_budget.json
```

Apply pattern-level `loss_weight`:

```bash
PYTHONPATH=src python scripts/apply_token_budget.py \
  --input data/generated/sft_correct_only_row_balanced_with_token_stats.csv \
  --stats data/generated/token_budget_row_balanced_stats.csv \
  --target-share configs/token_budget.json \
  --output data/generated/sft_correct_only_row_balanced_weighted.csv \
  --mode loss_weight \
  --pattern-col pattern \
  --token-col assistant_budget_units
```

Main output:

```text
data/generated/sft_correct_only_row_balanced_weighted.csv
```

The `loss_weight` column is used later during Kaggle training.

---

### 6. Create Final Training CSV

```bash
mkdir -p data/final

cp data/generated/sft_correct_only_row_balanced_weighted.csv \
  data/final/sft_train_final.csv

cp configs/token_budget.json \
  data/final/token_budget_used.json
```

Final CSV:

```text
data/final/sft_train_final.csv
```

---

### 7. Build Pre-tokenized Corpus

```bash
PYTHONPATH=src python scripts/build_tokenized_corpus.py \
  --input data/final/sft_train_final.csv \
  --output-dir data/final/tokenized_corpus_final
```

Output:

```text
data/final/tokenized_corpus_final/
├── train_00000.pt
├── train_00001.pt
├── ...
├── token_stats.csv
├── pattern_token_stats.csv
└── meta.json
```

This corpus is ready to upload as a Kaggle Dataset.

---

## Verify Final Corpus

Check the generated corpus metadata:

```bash
cat data/final/tokenized_corpus_final/meta.json
```

Inspect pattern-level token statistics:

```bash
PYTHONPATH=src python - <<'PY'
import pandas as pd

stats = pd.read_csv("data/final/tokenized_corpus_final/pattern_token_stats.csv")
print(stats.to_string(index=False))
PY
```

Check `.pt` shard structure:

```bash
PYTHONPATH=src python - <<'PY'
import torch
from pathlib import Path

path = Path("data/final/tokenized_corpus_final/train_00000.pt")
records = torch.load(path, map_location="cpu")

print("num records:", len(records))
print("keys:", records[0].keys())

for r in records[:5]:
    print({
        "id": r["id"],
        "pattern": r["pattern"],
        "rule_name": r["rule_name"],
        "seq_len": len(r["input_ids"]),
        "unmasked": sum(1 for x in r["labels"] if x != -100),
        "loss_weight": r.get("loss_weight"),
    })
PY
```

Expected keys:

```text
input_ids
attention_mask
labels
id
pattern
rule_name
loss_weight
```

---

## Inspect Solutions

You can inspect final solver outputs from the final CSV.

```bash
PYTHONPATH=src python - <<'PY'
import pandas as pd

df = pd.read_csv("data/final/sft_train_final.csv")

print("rows:", len(df))
print(df["pattern"].value_counts())

for pattern in df["pattern"].value_counts().index:
    row = df[df["pattern"] == pattern].sample(1, random_state=42).iloc[0]

    print("\n" + "#" * 100)
    print("pattern:", pattern)
    print("id:", row["id"])
    print("rule_name:", row.get("solver_rule_name", ""))
    print("answer:", row.get("answer", ""))
    print("solver_answer:", row.get("solver_answer", ""))
    print("solver_correct:", row.get("solver_correct", ""))
    print("loss_weight:", row.get("loss_weight", ""))

    solution = str(row.get("solver_solution", ""))
    print("\n[solver_solution preview]")
    print(solution[:3000])
PY
```

---

## Chat Template and Labels

The final tokenized corpus is generated from the `messages` column using the tokenizer chat template.

```text
messages
→ tokenizer.apply_chat_template(...)
→ input_ids / attention_mask / labels
```

When training from `tokenized_corpus_final`, do not apply the chat template again.

The labels follow assistant-only loss masking:

```text
user / prompt tokens:
  labels = -100

assistant solution / answer tokens:
  labels = token id
```

---

## Kaggle Dataset Upload

Zip the final corpus folder:

```bash
cd data/final
zip -r tokenized_corpus_final.zip tokenized_corpus_final
```

Upload `tokenized_corpus_final.zip` or the folder contents as a Kaggle Dataset.

Expected Kaggle path example:

```text
/kaggle/input/<dataset-name>/tokenized_corpus_final
```

---

## Kaggle Training

Training is intended to run on Kaggle GPU, not locally.

The Kaggle training script should:

* load `train_*.pt` shards
* use `input_ids`, `attention_mask`, and `labels` directly
* respect `labels=-100` masking
* use `loss_weight` if weighted training is enabled
* avoid re-applying the chat template

A weighted trainer implementation is provided in:

```text
src/nemotron_repro/training/weighted_trainer.py
```

---

## Useful Commands

### Check final CSV

```bash
PYTHONPATH=src python - <<'PY'
import pandas as pd

df = pd.read_csv("data/final/sft_train_final.csv")

print("rows:", len(df))
print(df["pattern"].value_counts())
print()
print(df.groupby("pattern")["loss_weight"].agg(["count", "mean", "min", "max"]))
PY
```

### Check final corpus files

```bash
find data/final/tokenized_corpus_final -maxdepth 1 -type f | sort
```

### Check generated files

```bash
find data -maxdepth 3 -type f | sort
```

---

## Notes

* Local preprocessing uses tokenizer only.
* Full model weights are loaded only during Kaggle training.
* If the model tokenizer or chat template changes, rebuild the tokenized corpus.
* `data/final/tokenized_corpus_final` is the main artifact for Kaggle training.
* `data/final/sft_train_final.csv` is the main human-readable final training dataset.
