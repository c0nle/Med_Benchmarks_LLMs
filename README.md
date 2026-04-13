# Med_Benchmarks_LLMs

A modular, resumable evaluation framework for benchmarking Large Language Models (LLMs) and Vision-Language Models (VLMs) on medical AI tasks — designed for HPC cluster environments with OpenAI-compatible inference servers (LiteLLM, vLLM, etc.).

## Benchmarks

| Benchmark | Type | Primary Metric | HuggingFace |
|-----------|------|----------------|-------------|
| [MedQA](#medqa) | Text MCQ | Accuracy | [openlifescienceai/medqa](https://huggingface.co/datasets/openlifescienceai/medqa) |
| [RaR](#rar) | Text MCQ | Accuracy | not on HF — local file required |
| [RadBench](#radbench) | Text MCQ | Accuracy | not on HF — local file required |
| [VQA-Med-2019](#vqa-med-2019) | Image VQA | BLEU + Accuracy | [simwit/vqa-med-2019](https://huggingface.co/datasets/simwit/vqa-med-2019) |
| [RadImageNet-VQA](#radimagenet-vqa) | Image VQA | Accuracy + LLM-Judge | [raidium/RadImageNet-VQA](https://huggingface.co/datasets/raidium/RadImageNet-VQA) |
| [Label Extraction](#label-extraction) | NER | Micro F1 | not on HF - local file required |
| [RadioRAG](#radiorag) | Text Open-ended QA | LLM-Judge Accuracy | not on HF - local file required |

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create your local config (excluded from git)
cp config.default.yaml config.yaml

# 3. Set your API key
export MED_SERVER_API_KEY="your_key"

# 4. Run
python main.py
```

Results are written to `results/`:
- `results/{benchmark}_results.csv` — raw model answers (one row per question)
- `results/{benchmark}_report.jsonl` — full evaluation report

---

## Configuration

Edit `config.yaml` (copied from `config.default.yaml`):

```yaml
server:
  url: "https://your-server:4000/v1"
  client: "openai_sdk"          # "openai_sdk" or "requests"
  model_name: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  verify_ssl: false
  api_key_env: "MED_SERVER_API_KEY"
  timeout_s: 60

benchmark: medqa                 # see below for all options

benchmark_settings:
  limit_samples: null            # null = full dataset, e.g. 100 for a quick test
  temperature: 0.0
  max_tokens: 256
  sleep_s: 0                     # delay between requests (rate limiting)
  max_errors: 200                # stop after N errors
```

### Benchmark Selection

```yaml
# Single benchmark
benchmark: medqa

# Multiple benchmarks (run sequentially, one result file each)
benchmark: [medqa, rar, radbench]

# All registered benchmarks
benchmark: all
```

---

## Running Benchmarks

```bash
# Run whatever is set in config.yaml
python main.py

# Quick test with 20 samples
# → set limit_samples: 20 in config.yaml, then:
python main.py
```

The run is **fully resumable** — if interrupted, re-run the same command. Already-processed rows are detected from the existing CSV and skipped.

---

## Evaluation

Evaluation runs automatically at the end of each benchmark. To re-run manually:

```bash
# MCQ benchmarks (MedQA, RaR, RadBench)
python evaluate.py results/medqa_results.csv --type mcq

# VQA benchmarks — automatic metrics (BLEU, exact match)
python evaluate.py results/vqa_med_2019_results.csv --type vqa

# VQA + LLM-as-a-Judge for open-ended questions (RadImageNet-VQA open subset)
# Requires a live LLM server configured in config.yaml
python evaluate.py results/radimagenet_vqa_results.csv --type vqa --judge

# Label Extraction
python evaluate.py results/label_extraction_results.csv --type extraction

# RadioRAG — automatic metrics only
python evaluate.py results/radiorag_results.csv --type open_qa

# RadioRAG — with LLM-as-a-Judge (primary metric, requires live LLM)
python evaluate.py results/radiorag_results.csv --type open_qa --judge
```

---

## Benchmark Details

### MedQA

**Task:** USMLE-style multiple-choice questions on general medical knowledge.  
**Metric:** Accuracy — rule-based letter extraction (A/B/C/D).  
**Paper:** Di Jin et al. (2021). *What disease does this patient have?* Applied Sciences. [[arXiv]](https://arxiv.org/abs/2009.13081) [[Journal]](https://www.mdpi.com/2076-3417/11/14/6421) [[GitHub]](https://github.com/jind11/MedQA)  
**Data:** Loaded automatically from [openlifescienceai/medqa](https://huggingface.co/datasets/openlifescienceai/medqa) or local parquet.

---

### RaR

**Task:** Radiology-specific MCQ requiring complex diagnostic reasoning over long patient histories.  
**Metric:** Accuracy — rule-based letter extraction.  
**Paper:** Wind et al. (2025). *Multi-step retrieval and reasoning improves radiology question answering.* npj Digital Medicine. [[Journal]](https://www.nature.com/articles/s41746-025-02250-5) [[arXiv]](https://arxiv.org/abs/2508.00743)  
**Data:** Not on HuggingFace — provide a local parquet file:

```bash
export RAR_PARQUET_PATH=/path/to/rar-test.parquet
```

---

### RadBench

**Task:** Clinical decision-making in radiology (377 closed-ended MCQ questions from MedPix and Radiopaedia cases).  
**Metric:** Accuracy — rule-based letter extraction.  
**Source:** Harrison.ai. *RadBench: Radiology Benchmark Framework.* [[Website]](https://harrison-ai.github.io/radbench/) [[GitHub]](https://github.com/harrison-ai/radbench)  
**Data:** Not on HuggingFace — download from the GitHub repo and provide:

```bash
export RADBENCH_PARQUET_PATH=/path/to/radbench.parquet
```

---

### VQA-Med-2019

**Task:** Visual Question Answering on medical images across four categories: Modality, Plane, Organ System, Abnormality. Requires a **vision-capable model (VLM)**.  
**Metrics:** BLEU (primary, sentence-level 1–4 gram) + Exact Match accuracy.  
**Paper:** Ben Abacha et al. (2019). *VQA-Med: Overview of the Medical Visual Question Answering Task at ImageCLEF 2019.* [[Paper]](https://ceur-ws.org/Vol-2380/paper_272.pdf) [[GitHub]](https://github.com/abachaa/VQA-Med-2019)  
**Data:** Loaded automatically from [simwit/vqa-med-2019](https://huggingface.co/datasets/simwit/vqa-med-2019) or local parquet:

```bash
export VQA_MED_2019_PARQUET_PATH=/path/to/vqa_med_2019.parquet
```

---

### RadImageNet-VQA

**Task:** VQA on CT, MRI, and X-ray images. Three task types with different evaluation protocols. Requires a **vision-capable model (VLM)**.

| Sub-task | Format | Metric |
|----------|--------|--------|
| Anatomy / Abnormality | Multiple-choice (A–D) | Accuracy (rule-based letter) |
| Existence / Comparison | Yes / No (closed) | Accuracy (exact match) |
| Pathology description | Open-ended | LLM-as-a-Judge (binary: correct/incorrect) |

**Paper:** Butsanets et al. (2025). *RadImageNet-VQA: A Large-Scale CT and MRI Dataset for Radiologic VQA.* [[arXiv]](https://arxiv.org/abs/2512.17396)  
**Data:** Loaded automatically from [raidium/RadImageNet-VQA](https://huggingface.co/datasets/raidium/RadImageNet-VQA) or local parquet:

```bash
export RADIMAGENET_VQA_PARQUET_PATH=/path/to/radimagenet_vqa.parquet
```

> **LLM-as-a-Judge:** Run automatically if you pass `--judge` to `evaluate.py`. The judge model is the same LLM configured in `config.yaml` and returns a binary 0/1 verdict per open-ended answer.

---

### Label Extraction

**Task:** Extract medical entities (findings, diagnoses, anatomical structures, pathologies) from free-text radiology reports.  
**Metric:** Micro F1 — TP/FP/FN aggregated across all instances before computing precision/recall (following the RadGraph evaluation protocol).  
**Reference:** Jain et al. (2021). *RadGraph: Extracting Clinical Entities and Relations from Radiology Reports.* NeurIPS Datasets & Benchmarks. [[arXiv]](https://arxiv.org/abs/2106.14463)  
**Data:** Not on HuggingFace (RadGraph requires PhysioNet credentials). Provide your own radiology report dataset:

```bash
export EXTRACTION_PARQUET_PATH=/path/to/extraction.parquet
# Required columns: "text" (report) + "entities" (comma-separated reference entities)
```

---

### RadioRAG

**Task:** Open-ended question answering on radiology cases across 18 subspecialties (chest, neuro, MSK, GI, pediatric, breast imaging, etc.). No image input — purely text-based clinical reasoning.  
**Metric:** LLM-as-a-Judge accuracy (binary: correct / incorrect, primary metric per paper). Human expert baseline: ~63%.  
**Dataset:** 104 questions — 80 from RSNA Case Collection (RSNA-RadioQA) + 24 expert-curated.  
**Paper:** Tayebi Arasteh et al. (2024). *RadioRAG: Online Retrieval-Augmented Generation for Radiology Question Answering.* Radiology: Artificial Intelligence. [[arXiv]](https://arxiv.org/abs/2407.15621) [[Journal]](https://pubs.rsna.org/doi/10.1148/ryai.240476) [[GitHub]](https://github.com/tayebiarasteh/RadioRAG)  
**Data:** Not on HuggingFace — download from GitHub and provide:

```bash
export RADIORAG_PARQUET_PATH=/path/to/radiorag.parquet
# Also accepts JSON: export RADIORAG_PARQUET_PATH=/path/to/radiorag.json
# Required columns: "question", "answer" (optional: "id", "subspecialty")
```

> **LLM-as-a-Judge:** Run automatically if you pass `--judge` to `evaluate.py`. The same LLM from `config.yaml` acts as judge and returns 0 (incorrect) or 1 (correct) per answer.

---

## Data Requirements Summary

| Benchmark | Available on HF | If not: env variable | Format |
|-----------|----------------|----------------------|--------|
| MedQA | ✅ auto | `MEDQA_PARQUET_PATH` | `question`, `choices` (dict), `answer` |
| RaR | ❌ manual | `RAR_PARQUET_PATH` | `question`, `choices` (dict), `answer` |
| RadBench | ❌ manual | `RADBENCH_PARQUET_PATH` | `question`, `choices` (dict), `gt` |
| VQA-Med-2019 | ✅ auto | `VQA_MED_2019_PARQUET_PATH` | `question`, `answer`, `image` |
| RadImageNet-VQA | ✅ auto | `RADIMAGENET_VQA_PARQUET_PATH` | `question`, `answer`, `image`, `question_type` |
| Label Extraction | ❌ manual | `EXTRACTION_PARQUET_PATH` | `text`, `entities` |
| RadioRAG | ❌ manual | `RADIORAG_PARQUET_PATH` | `question`, `answer` (parquet or JSON) |

---

## Project Structure

```
Med_Benchmarks_LLMs/
├── core/
│   └── client.py             # MedicalLLMClient — text + vision (base64 JPEG)
├── loaders/
│   ├── text_benchmarks.py    # MedQA, RaR, RadBench, Label Extraction loaders
│   └── vision_benchmarks.py  # VQA-Med-2019, RadImageNet-VQA loaders
├── tasks/
│   ├── mcq.py                # MCQ task runner (incremental, resumable)
│   ├── vqa.py                # VQA task runner (image + text, MCQ/yes-no/open)
│   └── extraction.py         # Label extraction task runner
├── scripts/
│   └── clean.py              # Remove __pycache__ / .pyc files
├── main.py                   # Entry point — single or multi-benchmark dispatch
├── evaluate.py               # All evaluation metrics
├── config.default.yaml       # Config template (commit this)
├── config.yaml               # Your config (git-ignored)
├── data/                     # Local dataset files (git-ignored)
└── results/                  # Output CSVs + JSONL reports (git-ignored)
```

---

## SSL / Self-signed Certificates

```yaml
server:
  verify_ssl: false           # disable verification (internal testing only)
  verify_ssl: "certs/ca.pem"  # provide CA bundle (recommended)
```

## HPC / Offline Environments

If `$HOME` is read-only, the framework creates a local HuggingFace cache at `.cache/huggingface/` automatically. Override:

```bash
export HF_HOME=/scratch/your_user/.cache/huggingface
```

For fully offline runs, download all datasets beforehand, export the parquet paths, and set `limit_samples` to a small value for a first test.

## Clean Bytecode Cache

```bash
python scripts/clean.py
```
