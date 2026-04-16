# Med_Benchmarks_LLMs

A modular, resumable evaluation framework for benchmarking LLMs and Vision-Language Models (VLMs) on medical AI tasks.

---

## Benchmarks

| Benchmark | Type | Requires VLM | Metric(s) |
|-----------|------|:---:|-----------|
| MedQA | Text MCQ (USMLE) | — | Accuracy |
| RaR | Text MCQ (Radiology board exam) | — | Accuracy |
| RadBench | X-ray image VQA | ✅ | MCQ Accuracy / Yes-No Accuracy / Open WBSS + BLEU-4 |
| VQA-Med-2019 | Medical image VQA | ✅ | Open BLEU-4 + WBSS + LLM-Judge |
| RadImageNet-VQA | CT/MRI/X-ray image VQA | ✅ | Open WBSS + BLEU-4 + LLM-Judge |
| Label Extraction | NER from radiology reports | — | Micro F1 |
| RadioRAG | Open-ended radiology QA | — | WBSS + BLEU-4 + LLM-Judge |

> **Text-only models** can still run on VLM benchmarks — they receive no image and results serve as a text-only baseline.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"
```

### 2. Create your config

```bash
cp config.default.yaml config.yaml
```

Edit `config.yaml`:

```yaml
server:
  url: "https://your-server:4000/v1"   # LiteLLM / vLLM / OpenAI-compatible endpoint
  model_name: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  api_key: "sk-your-key-here"          # or leave empty and set api_key_env
  verify_ssl: false
  timeout_s: 60

benchmark: medqa                        # see below for all options

benchmark_settings:
  limit_samples: 20                     # start small for a first test; null = full dataset
  temperature: 0.0
  max_tokens: 256
```

`config.yaml` is git-ignored — your credentials never leave your machine.

### 3. Download datasets

Most benchmarks load automatically. A few require manual download:

| Benchmark | Source | Place at |
|-----------|--------|----------|
| MedQA | Auto (HuggingFace) or [openlifescienceai/medqa](https://huggingface.co/datasets/openlifescienceai/medqa) | `data/medqa-test.parquet` |
| VQA-Med-2019 | Auto (HuggingFace) or [simwit/vqa-med-2019](https://huggingface.co/datasets/simwit/vqa-med-2019) | `data/vqa_med_2019.parquet` |
| RadImageNet-VQA | [raidium/RadImageNet-VQA](https://huggingface.co/datasets/raidium/RadImageNet-VQA) (see below) | `data/radimagenet_vqa_000.parquet` … |
| RadBench | [harrison-ai/radbench](https://github.com/harrison-ai/radbench) | `data/radbench.csv` |
| RaR | Contact authors via [paper](https://www.nature.com/articles/s41746-025-02250-5) | `data/rar-test.parquet` |
| RadioRAG | [tayebiarasteh/RadioRAG](https://github.com/tayebiarasteh/RadioRAG) | `data/radiorag.json` |
| Label Extraction | Your own radiology NER dataset | `data/extraction.parquet` |

Files placed at the paths above are detected automatically — no environment variables needed.

#### RadImageNet-VQA (large dataset — download once)

```bash
python3 -c "
from datasets import load_dataset
import os
os.makedirs('data', exist_ok=True)
ds = load_dataset('raidium/RadImageNet-VQA', name='alignment', split='train')
for i in range(0, len(ds), 50000):
    ds.select(range(i, min(i+50000, len(ds)))).to_parquet(f'data/radimagenet_vqa_{i//50000:03d}.parquet')
print('Done.')
"
```

#### RadBench images (X-rays)

```bash
git clone https://github.com/harrison-ai/radbench data/radbench_repo
cp data/radbench_repo/data/radbench/radbench.csv data/radbench.csv
rm -rf data/radbench_repo

# Download X-ray images (~68 Radiopaedia images, MedPix unavailable)
python3 scripts/download_radbench_images.py
```

> MedPix images cannot be downloaded automatically (API discontinued). The loader skips those 212 questions automatically and only evaluates the 285 Radiopaedia questions that have images.

### 4. Run

```bash
python main.py
```

---

## Running Benchmarks

```bash
# Run what is set in config.yaml
python main.py
```

The run is **fully resumable** — if interrupted, re-run the same command. Already-processed rows are skipped automatically.

Progress is printed every 50 questions:
```
  [ 50/285]  17%  7.3 q/s  ETA 00:32  errors: 0
```

At the end a summary is printed and a timestamped log is written to `results/run_<timestamp>.log`:
```
============================================================
  RESULTS
============================================================
  medqa                  accuracy_pct: 79.89%
  radbench               mcq_accuracy_pct: 47.62%  yes_no_accuracy_pct: 58.11%  open_wbss_pct: 65.78%
  vqa_med_2019           open_wbss_pct: 50.44%  open_judge_accuracy_pct: 48.60%
  radimagenet_vqa        open_wbss_pct: 57.40%  open_judge_accuracy_pct: 28.04%
============================================================
```

### Selecting benchmarks

```yaml
# Single benchmark
benchmark: medqa

# Multiple benchmarks
benchmark: [medqa, radbench, vqa_med_2019]

# All registered benchmarks
benchmark: all
```

Available names: `medqa`, `rar`, `radbench`, `vqa_med_2019`, `radimagenet_vqa`, `label_extraction`, `radiorag`

### Limiting samples (for quick tests)

```yaml
benchmark_settings:
  limit_samples: 100   # null = full dataset
```

---

## Output Files

| File | Contents |
|------|----------|
| `results/{benchmark}_results.csv` | Raw model answers, one row per question |
| `results/{benchmark}_report.jsonl` | Full evaluation report with all metrics |
| `results/run_<timestamp>.log` | Full run log including verbose per-question output |

---

## Re-running Evaluation

Evaluation runs automatically after each benchmark. To re-evaluate existing results manually:

```bash
# MCQ benchmarks (MedQA, RaR)
python evaluate.py results/medqa_results.csv --type mcq

# VQA benchmarks (RadBench, VQA-Med-2019, RadImageNet-VQA)
python evaluate.py results/radbench_results.csv --type vqa

# VQA + LLM-as-a-Judge for open-ended questions
python evaluate.py results/radimagenet_vqa_results.csv --type vqa --judge

# Label Extraction
python evaluate.py results/label_extraction_results.csv --type extraction

# RadioRAG + LLM-as-a-Judge (primary metric)
python evaluate.py results/radiorag_results.csv --type open_qa --judge
```

---

## Metrics

### Accuracy
Rule-based letter extraction (A/B/C/D) for MCQ; exact match for Yes/No. Random baseline: 25% (4-choice MCQ), 50% (Yes/No).

### WBSS — Word-Based Semantic Similarity
Measures semantic similarity between model answer and reference answer using Wu-Palmer similarity from WordNet. Gives partial credit for synonyms and paraphrases.

| WBSS | Interpretation |
|------|----------------|
| < 30% | Poor |
| 30–50% | Moderate |
| 50–70% | Good — correct domain, different wording |
| > 70% | Strong |

### BLEU-4
Per-item BLEU-4 using the VQA-Med-2019 official preprocessing (Ben Abacha et al., ImageCLEF 2019): lowercase → strip punctuation → NLTK word tokenize → remove English stopwords → Snowball stemming. Average over all items. Primary metric for VQA-Med-2019; reported alongside WBSS for RadioRAG and RadImageNet-VQA.

Note: the word "no" is an NLTK English stopword, so Yes/No answers of "no" score 0 even when correct — this is a known quirk of the official evaluator, reproduced exactly here.

### Micro F1
Used for Label Extraction. TP/FP/FN aggregated across all items before computing precision/recall (following RadGraph protocol).

### LLM-as-a-Judge
A second LLM (configured in `config.yaml` under `judge:`) scores each open-ended answer as 0 or 1. Requires `--judge` flag or `run_judge: true`. Human expert baseline on RadioRAG: ~63%.

---

## Reference Results (google/gemma-4-26B-A4B-it)

| Benchmark | Metric | Score |
|-----------|--------|-------|
| MedQA | Accuracy | 79.89% |
| RadBench (Radiopaedia only, n=285) | MCQ Accuracy | 47.62% |
| RadBench | Yes-No Accuracy | 58.11% |
| RadBench | Open WBSS | 65.78% |
| VQA-Med-2019 | Open WBSS | 50.44% |
| VQA-Med-2019 | LLM-Judge | 48.60% |
| RadImageNet-VQA (n=5000) | Open WBSS | 57.40% |
| RadImageNet-VQA | LLM-Judge | 28.04% |

---

## Project Structure

```
Med_Benchmarks_LLMs/
├── core/
│   ├── client.py             # LLM client — text + vision (base64)
│   └── logger.py             # RunLogger — tees stdout to log file
├── loaders/
│   ├── text_benchmarks.py    # MedQA, RaR, Label Extraction, RadioRAG
│   └── vision_benchmarks.py  # RadBench, VQA-Med-2019, RadImageNet-VQA
├── tasks/
│   ├── mcq.py                # MCQ task runner
│   ├── vqa.py                # VQA task runner (image + text)
│   ├── extraction.py         # Label extraction runner
│   └── open_qa.py            # Open-ended QA runner
├── scripts/
│   └── download_radbench_images.py   # Download RadBench X-ray images
├── main.py                   # Entry point
├── evaluate.py               # All evaluation metrics
├── config.default.yaml       # Config template (tracked in git)
├── config.yaml               # Your config with credentials (git-ignored)
├── data/                     # Local dataset files (git-ignored)
└── results/                  # Output CSVs + reports (git-ignored)
```

---

## HPC / Offline Environments

If `$HOME` is read-only (common on HPC clusters), set:

```bash
export HF_HOME=/path/to/writable/dir/.cache/huggingface
```

The framework will detect a read-only `$HOME` and fall back to `.cache/huggingface/` inside the project directory automatically.

For fully offline runs, download all datasets beforehand and place them in `data/` as described above.

---

## SSL / Self-signed Certificates

```yaml
server:
  verify_ssl: false           # disable (internal/testing only)
  verify_ssl: "certs/ca.pem"  # provide CA bundle (recommended for production)
```
