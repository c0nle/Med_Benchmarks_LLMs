# Med_Benchmarks_LLMs

A modular, resumable evaluation framework for benchmarking LLMs and Vision-Language Models (VLMs) on medical AI tasks.

---

## Benchmarks

| Benchmark | Type | Requires VLM | Metric(s) |
|-----------|------|:---:|-----------|
| MedQA | Text MCQ (USMLE) | — | Accuracy |
| RaR | Text MCQ (Radiology board exam) | — | Accuracy |
| RadBench | X-ray image VQA | ✅ | MCQ Accuracy / Yes-No Accuracy / Open WBSS |
| VQA-Med-2019 | Medical image VQA | ✅ | Open WBSS + LLM-Judge |
| RadImageNet-VQA | CT/MRI/X-ray image VQA | ✅ | Open WBSS + LLM-Judge |
| Label Extraction | NER from radiology reports | — | Micro F1 |
| RadioRAG | Text MCQ (Radiology) | — | Accuracy |

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
| RaR | Contact authors via [paper](https://www.nature.com/articles/s41746-025-02250-5) | `data/RaR_dataset_WithAnswer.csv` |
| RadioRAG | Contact authors via [GitHub](https://github.com/tayebiarasteh/RadioRAG) | `data/RadioRAG_WithOptions_WithAnswer.csv` |
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

## Benchmark Interpretation

| Benchmark | What it measures | Metric | Random baseline | Notes |
|-----------|-----------------|--------|:-:|-------|
| MedQA | USMLE Step 1–3 clinical reasoning (4-choice MCQ) | Accuracy | 25% | General medicine knowledge |
| RaR | Radiology board-style MCQ (5-choice) | Accuracy | 20% | Domain-specific radiology reasoning |
| RadioRAG | Radiology factual QA (4-choice MCQ) | Accuracy | 25% | Originally open-ended; converted to MCQ |
| RadBench MCQ | Chest X-ray clinical reasoning with image (4-choice) | Accuracy | 25% | Requires VLM; Radiopaedia cases only (n=285) |
| RadBench Yes/No | Binary image questions (e.g. "Is there a pneumothorax?") | Accuracy | 50% | Requires VLM |
| RadBench Open | Free-text description of X-ray findings | WBSS | — | Semantic similarity; > 60% is good |
| VQA-Med-2019 | Medical image VQA: modality, organ, plane, abnormality | WBSS / LLM-Judge | — | 500-item validation set (ImageCLEF 2019) |
| RadImageNet-VQA | CT/MRI pathology description (open), binary (yes/no), MCQ | WBSS / LLM-Judge / Accuracy | 25–50% | 9K-item benchmark split; 1K images |
| Label Extraction | Entity extraction from radiology reports (NER) | Micro F1 | — | Higher = more complete entity set |

---

## Metrics

### Accuracy
Rule-based letter extraction (A/B/C/D) for MCQ; exact match for Yes/No. Random baseline: 25% (4-choice MCQ), 20% (5-choice), 50% (Yes/No).

### WBSS — Word-Based Semantic Similarity
Measures semantic similarity between model answer and reference answer using Wu-Palmer similarity from WordNet. Gives partial credit for synonyms and paraphrases.

| WBSS | Interpretation |
|------|----------------|
| < 30% | Poor |
| 30–50% | Moderate |
| 50–70% | Good — correct domain, different wording |
| > 70% | Strong |

### Micro F1
Used for Label Extraction. TP/FP/FN aggregated across all items before computing precision/recall (following RadGraph protocol).

### LLM-as-a-Judge
A second LLM (configured in `config.yaml` under `judge:`) scores each open-ended answer as 0 or 1. Requires `--judge` flag or `run_judge: true`.

---

## Citations

If you use this framework or the underlying datasets in your work, please cite the original sources.

**Datasets**

- **MedQA**: Jin et al. (2021). *What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams.* Applied Sciences. https://arxiv.org/abs/2009.13081
- **VQA-Med-2019**: Ben Abacha et al. (2019). *VQA-Med: Overview of the Medical Visual Question Answering Task at ImageCLEF 2019.* CLEF 2019. https://www.imageclef.org/2019/medical/vqa
- **RadImageNet-VQA**: Butsanets et al. (2025). *RadImageNet-VQA.* https://huggingface.co/datasets/raidium/RadImageNet-VQA
- **RadBench**: Harrison.ai (2024). *RadBench: Benchmarking Large Language Models for Radiology.* https://github.com/harrison-ai/radbench
- **RaR**: Contact authors via https://www.nature.com/articles/s41746-025-02250-5
- **RadioRAG**: Tayebi Arasteh et al. (2024). *RadioRAG: Factual Large Language Models for Enhanced Diagnostics in Radiology Using Dynamic Retrieval Augmented Generation.* https://github.com/tayebiarasteh/RadioRAG
- **Label Extraction / RadGraph**: Jain et al. (2021). *RadGraph: Extracting Clinical Entities and Relations from Radiology Reports.* NeurIPS 2021. https://physionet.org/content/radgraph/

**Evaluation Methodology**

- **WBSS**: Ben Abacha et al. (2019), see VQA-Med-2019 above.
- **LLM-as-a-Judge**: Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023. https://arxiv.org/abs/2306.05685

