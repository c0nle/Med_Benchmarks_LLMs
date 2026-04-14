# Med_Benchmarks_LLMs

A modular, resumable evaluation framework for benchmarking Large Language Models (LLMs) and Vision-Language Models (VLMs) on medical AI tasks — designed for HPC cluster environments with OpenAI-compatible inference servers (LiteLLM, vLLM, etc.).

## Benchmarks

| Benchmark | Type | Model Required | Metric(s) Reported |
|-----------|------|----------------|--------------------|
| [MedQA](#medqa) | Text MCQ | Text LLM | Accuracy |
| [RaR](#rar) | Text MCQ | Text LLM | Accuracy |
| [RadBench](#radbench) | VLM (X-ray images) | **VLM** | MCQ Accuracy / Yes-No Accuracy / Open WBSS |
| [VQA-Med-2019](#vqa-med-2019) | Image VQA | **VLM** | Open WBSS |
| [RadImageNet-VQA](#radimagenet-vqa) | Image VQA | **VLM** | MCQ Accuracy / Yes-No Accuracy / Open WBSS |
| [Label Extraction](#label-extraction) | NER (text) | Text LLM | Micro F1 |
| [RadioRAG](#radiorag) | Text Open-ended QA | Text LLM | WBSS / LLM-Judge |

---

## Which Model Do I Need?

| If you want to run… | You need… |
|---------------------|-----------|
| MedQA, RaR, Label Extraction, RadioRAG | Any **text-only LLM** (e.g. Llama-3, Mistral, GPT-4o-text) |
| RadBench, VQA-Med-2019, RadImageNet-VQA | A **Vision-Language Model (VLM)** with image input support (e.g. LLaVA, GPT-4o, Claude 3) |
| All benchmarks together | A VLM — it handles both text and image inputs |

> Text-only models will still run on VLM benchmarks but receive no image; treat results as a text-only baseline only.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"

# 2. Log in to HuggingFace (required for gated datasets)
huggingface-cli login

# 3. Create your local config (excluded from git)
cp config.default.yaml config.yaml

# 4. Set your API key
export MED_SERVER_API_KEY="your_key"

# 5. Run
python main.py
```

Results are written to `results/`:
- `results/{benchmark}_results.csv` — raw model answers (one row per question)
- `results/{benchmark}_report.jsonl` — full evaluation report

---

## Local Setup Step by Step

### 1. Clone and install

```bash
git clone <repo-url>
cd Med_Benchmarks_LLMs
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"
```

### 2. Configure your inference server

Copy the template and edit:

```bash
cp config.default.yaml config.yaml
```

```yaml
server:
  url: "https://your-server:4000/v1"   # LiteLLM / vLLM / OpenAI endpoint
  client: "openai_sdk"                  # "openai_sdk" (recommended) or "requests"
  model_name: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  verify_ssl: false                     # set to true or a CA path for production
  api_key_env: "MED_SERVER_API_KEY"
  timeout_s: 60

benchmark: medqa                        # or a list, or "all"

benchmark_settings:
  limit_samples: 20                     # start with 20 for a quick sanity-check
  temperature: 0.0
  max_tokens: 256
  sleep_s: 0
  max_errors: 200
```

### 3. Set credentials

```bash
export MED_SERVER_API_KEY="your_key"

# HuggingFace login (required for raidium/RadImageNet-VQA and similar gated datasets)
huggingface-cli login
# or: export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

### 4. Download datasets that are not on HuggingFace

| Benchmark | Where to get it | env variable |
|-----------|----------------|--------------|
| RaR (n=65) | Contact authors via [npj paper](https://www.nature.com/articles/s41746-025-02250-5) | `RAR_PARQUET_PATH` |
| RadBench | [harrison-ai/radbench](https://github.com/harrison-ai/radbench) | `RADBENCH_PARQUET_PATH` |
| Label Extraction | Your own radiology NER dataset | `EXTRACTION_PARQUET_PATH` |
| RadioRAG | [tayebiarasteh/RadioRAG](https://github.com/tayebiarasteh/RadioRAG) | `RADIORAG_PARQUET_PATH` |

```bash
export RAR_PARQUET_PATH=/path/to/rar.parquet
export RADBENCH_PARQUET_PATH=/path/to/radbench.parquet
export RADIORAG_PARQUET_PATH=/path/to/radiorag.json   # also accepts .parquet
```

### 5. Run a quick test

```bash
# config.yaml: limit_samples: 20, benchmark: medqa
python main.py
```

### 6. Run full evaluation

```bash
# config.yaml: limit_samples: null, benchmark: [medqa, rar, radiorag]
python main.py
```

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
benchmark: [medqa, rar, radiorag]

# All registered benchmarks
benchmark: all
```

---

## Running Benchmarks

```bash
# Run whatever is set in config.yaml
python main.py
```

The run is **fully resumable** — if interrupted, re-run the same command. Already-processed rows are detected from the existing CSV and skipped.

Progress is printed every 50 questions:
```
  [ 50/500]  10%  1.3 q/s  ETA 06:24  errors: 0
```

At the end, a results summary is always printed:
```
============================================================
  RESULTS
============================================================
  medqa                  accuracy_pct: 80.44%
  vqa_med_2019           open_wbss_pct: 54.10%
============================================================
```

---

## Evaluation

Evaluation runs automatically after each benchmark. To re-run manually:

```bash
# MCQ benchmarks (MedQA, RaR)
python evaluate.py results/medqa_results.csv --type mcq

# VQA benchmarks (RadBench, VQA-Med-2019, RadImageNet-VQA)
python evaluate.py results/vqa_med_2019_results.csv --type vqa

# VQA + LLM-as-a-Judge for open-ended questions
# Requires a live LLM server configured in config.yaml
python evaluate.py results/radimagenet_vqa_results.csv --type vqa --judge

# Label Extraction
python evaluate.py results/label_extraction_results.csv --type extraction

# RadioRAG
python evaluate.py results/radiorag_results.csv --type open_qa

# RadioRAG + LLM-as-a-Judge (primary metric)
python evaluate.py results/radiorag_results.csv --type open_qa --judge
```

---

## Metrics Explained

### Accuracy

Used for MCQ and Yes/No questions. The model's answer is extracted (rule-based letter parser for MCQ, exact match for Yes/No) and compared to the reference.

- For MCQ: the model must reply with a letter (A/B/C/D). The parser extracts the first unambiguous letter from the response.
- For Yes/No: the answer is normalised (lowercase, punctuation removed) and compared directly.

A random baseline is 25% for 4-choice MCQ and 50% for Yes/No.

### WBSS — Word-Based Semantic Similarity

Used for all open-ended answers. Measures **semantic similarity** between the model's answer and the reference answer using Wu-Palmer similarity scores from WordNet.

Unlike exact-match metrics, WBSS gives partial credit when the model uses synonyms or paraphrases the correct answer. For example:

| Model answer | Reference | WBSS | Notes |
|---|---|---|---|
| `"CT scan"` | `"ct angiography"` | ~0.65 | Semantically related |
| `"X-ray"` | `"xr - plain film"` | ~0.37 | Partially related |
| `"Ultrasound"` | `"ultrasound"` | 1.0 | Exact semantic match |
| `"MRI"` | `"CT scan"` | ~0.30 | Different modalities |

**Score range:** 0.0 (completely unrelated) to 1.0 (identical meaning)

| WBSS | Interpretation |
|------|----------------|
| < 30% | Poor — answers are semantically unrelated |
| 30–50% | Moderate — partial overlap in meaning |
| 50–70% | Good — model understands the domain but may use different vocabulary |
| > 70% | Strong — answers are semantically equivalent |

WBSS requires NLTK WordNet: `nltk.download('wordnet')`.

### Micro F1

Used for Label Extraction. TP/FP/FN are aggregated across all instances before computing precision and recall (following the RadGraph evaluation protocol). This avoids the instability of per-instance F1 on short lists.

| Micro F1 | Interpretation |
|----------|----------------|
| < 30% | Poor extraction quality |
| 30–60% | Moderate — misses many entities or hallucinates |
| 60–80% | Good — usable for downstream tasks |
| > 80% | Strong — close to supervised NER (RadGraph baseline: ~82%) |

### LLM-as-a-Judge

Used optionally for open-ended benchmarks (RadioRAG, RadImageNet-VQA open subset). The same LLM configured in `config.yaml` acts as a judge and receives:
- the original question
- the ground-truth answer
- the model's answer

It returns `1` (correct) or `0` (incorrect). The final score is the fraction of correct answers.

To enable it, pass `--judge` to `evaluate.py` or set `run_judge=True` in code. It requires a live LLM server and makes one additional call per open-ended question.

| LLM-Judge Score | Interpretation |
|---|---|
| < 40% | Below human baseline |
| ~63% | Human expert baseline on RadioRAG (per paper) |
| > 70% | Strong performance |

---

## Interpreting Results

### MedQA (Accuracy)

| Score | Interpretation |
|-------|---------------|
| < 50% | Below chance — likely prompt or parsing issue |
| 50–60% | Weak |
| 60–70% | Moderate; approaches USMLE passing threshold (~60%) |
| 70–80% | Good |
| > 80% | Strong; comparable to top human performance |

Reference: GPT-4 ~87%, Llama-3 70B ~78%, Llama-3 8B ~60%.

### VQA-Med-2019 / RadBench / RadImageNet-VQA (WBSS)

The reference answers in VQA-Med-2019 use internal shortcodes (`"iv"`, `"xr - plain film"`, `"cta - ct angiography"`) that were used by the 2019 classification systems. A general LLM produces expanded natural language (`"intravenous contrast"`, `"X-ray"`, `"CT angiography"`), which is semantically correct but doesn't match the shortcodes exactly.

WBSS captures this semantic correctness. A score of 50–60% means the model's answers are in the right ballpark medically, even if phrased differently. The top 2019 systems achieved ~68% WBSS.

For LLM-Judge accuracy on open-ended subsets, see the [LLM-as-a-Judge](#llm-as-a-judge) section above.

### RadioRAG (WBSS / LLM-Judge)

WBSS gives a quick automatic estimate. The primary metric per paper is LLM-Judge accuracy. Human expert baseline: ~63%.

---

## Benchmark Details

### MedQA

**Task:** USMLE-style multiple-choice questions on general medical knowledge.  
**Metric:** Accuracy — rule-based letter extraction (A/B/C/D).  
**Paper:** Di Jin et al. (2021). *What disease does this patient have?* Applied Sciences. [[arXiv]](https://arxiv.org/abs/2009.13081) [[GitHub]](https://github.com/jind11/MedQA)  
**Data:** Loaded automatically from [openlifescienceai/medqa](https://huggingface.co/datasets/openlifescienceai/medqa) or local parquet:

```bash
export MEDQA_PARQUET_PATH=/path/to/medqa-test.parquet
```

---

### RaR

**Task:** Radiology board-exam MCQ requiring complex multi-step diagnostic reasoning over detailed patient histories. 65 questions, 5 answer choices.  
**Metric:** Accuracy — rule-based letter extraction.  
**Paper:** Wind et al. (2025). *Multi-step retrieval and reasoning improves radiology question answering.* npj Digital Medicine. [[Journal]](https://www.nature.com/articles/s41746-025-02250-5)  
**Data:** Not on HuggingFace — contact the authors via the paper link and provide:

```bash
export RAR_PARQUET_PATH=/path/to/rar-test.parquet
# Required columns: "question", "options"/"choices", "answer"
```

---

### RadBench

**Task:** Clinical decision-making using **real X-ray images** (89 cases from MedPix and Radiopaedia). Requires a **vision-capable model (VLM)**. 497 questions total: 377 closed-ended (MCQ / yes-no) + 120 open-ended.

| Sub-task | Metric |
|----------|--------|
| Closed-ended MCQ | Letter accuracy (rule-based) |
| Closed-ended Yes/No | Exact match accuracy |
| Open-ended | WBSS + optional LLM-Judge |

**Source:** Harrison.ai. *RadBench: Radiology Benchmark Framework.* [[Website]](https://harrison-ai.github.io/radbench/) [[GitHub]](https://github.com/harrison-ai/radbench)  
**Data:** Not on HuggingFace — download from the GitHub repository and provide:

```bash
export RADBENCH_PARQUET_PATH=/path/to/radbench.parquet
# Expected columns: QUESTION, ANSWER, A_TYPE, OPTIONS, CASE_ID, IMAGE_ORGAN, modality
# Images (X-rays) are stored separately in /images/ — without images, model runs text-only
```

---

### VQA-Med-2019

**Task:** Visual Question Answering on medical images across four categories: Modality, Plane, Organ System, Abnormality. Requires a **vision-capable model (VLM)**.  
**Metric:** WBSS (semantic similarity). Optional LLM-Judge via `--judge`.  
**Paper:** Ben Abacha et al. (2019). *VQA-Med: Overview of the Medical Visual Question Answering Task at ImageCLEF 2019.* [[Paper]](https://ceur-ws.org/Vol-2380/paper_272.pdf) [[GitHub]](https://github.com/abachaa/VQA-Med-2019)  
**Data:** Loaded automatically from [simwit/vqa-med-2019](https://huggingface.co/datasets/simwit/vqa-med-2019) or local parquet:

```bash
export VQA_MED_2019_PARQUET_PATH=/path/to/vqa_med_2019.parquet
# Required columns: "question", "answer", "image"
```

---

### RadImageNet-VQA

**Task:** VQA on CT, MRI, and X-ray images. Three task types with different evaluation protocols. Requires a **vision-capable model (VLM)**.

| Sub-task | Format | Metric |
|----------|--------|--------|
| Anatomy / Abnormality | Multiple-choice (A–D) | Accuracy (rule-based letter) |
| Existence / Comparison | Yes / No (closed) | Accuracy (exact match) |
| Pathology description | Open-ended | WBSS + optional LLM-Judge |

**Paper:** Butsanets et al. (2025). *RadImageNet-VQA: A Large-Scale CT and MRI Dataset for Radiologic VQA.* [[arXiv]](https://arxiv.org/abs/2512.17396)  
**Data:** Loaded automatically from [raidium/RadImageNet-VQA](https://huggingface.co/datasets/raidium/RadImageNet-VQA) (gated — requires HuggingFace login) or local parquet:

```bash
huggingface-cli login   # one-time setup

# or save locally after first download:
python3 -c "
from datasets import load_dataset
ds = load_dataset('raidium/RadImageNet-VQA', split='test')
ds.to_parquet('data/radimagenet_vqa.parquet')
"
export RADIMAGENET_VQA_PARQUET_PATH=data/radimagenet_vqa.parquet
```

---

### Label Extraction

**Task:** Extract medical entities (findings, diagnoses, anatomical structures, pathologies) from free-text radiology reports.  
**Metric:** Micro F1 — TP/FP/FN aggregated across all instances before computing precision/recall (following the RadGraph evaluation protocol).  
**Reference:** Jain et al. (2021). *RadGraph: Extracting Clinical Entities and Relations from Radiology Reports.* NeurIPS Datasets & Benchmarks. [[arXiv]](https://arxiv.org/abs/2106.14463)  
**Data:** Not on HuggingFace. Provide your own radiology report dataset:

```bash
export EXTRACTION_PARQUET_PATH=/path/to/extraction.parquet
# Required columns: "text" (report) + "entities" (comma-separated reference entities)
```

---

### RadioRAG

**Task:** Open-ended question answering on radiology cases across 18 subspecialties (chest, neuro, MSK, GI, pediatric, breast imaging, etc.). No image input — purely text-based clinical reasoning.  
**Metric:** WBSS (automatic) + LLM-Judge accuracy (primary, requires `--judge`). Human expert baseline: ~63%.  
**Dataset:** 104 questions — 80 from RSNA Case Collection (RSNA-RadioQA) + 24 expert-curated.  
**Paper:** Tayebi Arasteh et al. (2024). *RadioRAG: Online Retrieval-Augmented Generation for Radiology Question Answering.* Radiology: Artificial Intelligence. [[arXiv]](https://arxiv.org/abs/2407.15621) [[GitHub]](https://github.com/tayebiarasteh/RadioRAG)  
**Data:** Not on HuggingFace — download from GitHub and provide:

```bash
export RADIORAG_PARQUET_PATH=/path/to/radiorag.json   # also accepts .parquet
# Required columns: "question", "answer" (optional: "id", "subspecialty")
```

---

## Data Requirements Summary

| Benchmark | Available on HF | env variable | Required columns |
|-----------|----------------|--------------|-----------------|
| MedQA | ✅ auto | `MEDQA_PARQUET_PATH` | `question`, `options` (dict), `answer` |
| RaR | ❌ manual | `RAR_PARQUET_PATH` | `question`, `options`/`choices`, `answer` |
| RadBench | ❌ manual | `RADBENCH_PARQUET_PATH` | `QUESTION`, `ANSWER`, `A_TYPE`, `OPTIONS` |
| VQA-Med-2019 | ✅ auto | `VQA_MED_2019_PARQUET_PATH` | `question`, `answer`, `image` |
| RadImageNet-VQA | ✅ auto (gated) | `RADIMAGENET_VQA_PARQUET_PATH` | `question`, `answer`, `image` |
| Label Extraction | ❌ manual | `EXTRACTION_PARQUET_PATH` | `text`, `entities` |
| RadioRAG | ❌ manual | `RADIORAG_PARQUET_PATH` | `question`, `answer` (parquet or JSON) |

---

## Project Structure

```
Med_Benchmarks_LLMs/
├── core/
│   └── client.py             # MedicalLLMClient — text + vision (base64 JPEG)
├── loaders/
│   ├── text_benchmarks.py    # MedQA, RaR, Label Extraction, RadioRAG loaders
│   └── vision_benchmarks.py  # RadBench, VQA-Med-2019, RadImageNet-VQA loaders
├── tasks/
│   ├── mcq.py                # MCQ task runner (MedQA, RaR) — incremental, resumable
│   ├── vqa.py                # VQA task runner (RadBench, VQA-Med-2019, RadImageNet-VQA)
│   ├── extraction.py         # Label extraction task runner
│   └── open_qa.py            # Open-ended QA runner (RadioRAG)
├── main.py                   # Entry point — single or multi-benchmark dispatch
├── evaluate.py               # All evaluation metrics (MCQ, VQA, NER, open QA)
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
