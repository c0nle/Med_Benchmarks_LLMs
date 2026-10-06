# Med_Benchmarks_LLMs

A modular, resumable evaluation framework for benchmarking LLMs and Vision-Language Models (VLMs) on medical AI tasks.

---

## Benchmarks

| Benchmark (config name)                  | Type                                         | Requires VLM | Metric(s)                                                       |
| ---------------------------------------- | -------------------------------------------- | :----------: | --------------------------------------------------------------- |
| MedQA (`medqa`)                          | Text MCQ (USMLE, 4 options)                  |      —       | Accuracy                                                        |
| RaR (`rar`)                              | Text MCQ (radiology, 5 options)              |      —       | Accuracy                                                        |
| RadioRAG (`radiorag`)                    | Text MCQ variant (radiology, 4 options)      |      —       | Accuracy                                                        |
| RadBench (`radbench`)                    | X-ray image VQA (MCQ / yes-no / open)        |      ✅      | MCQ Acc. / Yes-No Acc. / Open: Exact, WBSS, LLM-Judge           |
| VQA-Med-2019 (`vqa_med_2019`)            | Medical image VQA (open)                     |      ✅      | Exact match, WBSS, LLM-Judge                                    |
| RadImageNet-VQA (`radimagenet_vqa`)      | CT/MRI image VQA (MCQ / yes-no / open)       |      ✅      | MCQ Acc. / Yes-No Acc. / Open: Exact, WBSS, LLM-Judge           |
| Mamma-MRT extraction (`label_extraction_mamma`) | Structured extraction, German breast MRI reports | — | Accuracy + Macro-F1 per field; lesion-type Micro-F1 / Exact; 95% CIs |
| Arm X-ray extraction (`label_extraction_arm`)   | Binary labels + citation, German arm X-ray reports | — | Micro-/Macro-F1 (overall and per region), Sens./Spec. per label, citation match; 95% CI |
| Label Extraction (`label_extraction`)    | NER (RadGraph-style)                         |      —       | Micro F1 (needs `data/extraction.parquet`, not included)        |

> VLM benchmarks always send the image(s), so the model must accept image input.
> Multi-image RadBench questions send all images of the case.

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
  timeout_s: 300                       # long JSON answers (label extraction) need > 60 s

benchmark: medqa                        # see below for all options

benchmark_settings:
  limit_samples: 20                     # start small for a first test; null = full dataset
  temperature: 0.0
  max_tokens: 256
```

`config.yaml` is git-ignored.

### 3. Download datasets

Most benchmarks load automatically. A few require manual download:

| Benchmark        | Source                                                                                               | Place at                                     |
| ---------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| MedQA            | Auto (HuggingFace) or[openlifescienceai/medqa](https://huggingface.co/datasets/openlifescienceai/medqa) | `data/medqa-test.parquet`                  |
| VQA-Med-2019     | Auto (HuggingFace) or[simwit/vqa-med-2019](https://huggingface.co/datasets/simwit/vqa-med-2019)         | `data/vqa_med_2019.parquet`                |
| RadImageNet-VQA  | [raidium/RadImageNet-VQA](https://huggingface.co/datasets/raidium/RadImageNet-VQA) (see below)          | `data/radimagenet_vqa_benchmark.parquet`   |
| RadBench         | [harrison-ai/radbench](https://github.com/harrison-ai/radbench)                                         | `data/radbench.csv`                        |
| RaR              | Contact authors via[paper](https://www.nature.com/articles/s41746-025-02250-5)                          | `data/RaR_dataset_WithAnswer.csv`          |
| RadioRAG         | Contact authors via[GitHub](https://github.com/tayebiarasteh/RadioRAG)                                  | `data/RadioRAG_WithOptions_WithAnswer.csv` |
| Mamma-MRT        | Not public (local patient data)                                                                      | `data/label_extraction/label_extraction_gt.xlsx` + `hiwi_gt_ergaenzung.xlsx` |
| Arm X-ray        | Not public (local patient data; Kreutzer et al., Eur Radiol 2025)                                    | `data/label_extraction/Label_Extraction_Kilian/` |
| Label Extraction | Your own radiology NER dataset                                                                       | `data/extraction.parquet`                  |

Files placed at the paths above are detected automatically.

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

Command-line options (override `config.yaml`):

```bash
python main.py --limit 5                          # quick test, 5 items per benchmark
python main.py --benchmark medqa,radbench         # select benchmarks
python main.py --run-dir results/run_<timestamp>  # resume: finished items are skipped,
                                                  # failed requests and missing judge verdicts are redone
```

On the HPC cluster submit `run.sh` (arguments are passed to `main.py`):

```bash
sbatch run.sh --limit all
sbatch run.sh --run-dir results/run_<timestamp> --limit all   # resume after a crash/timeout
```

Transient server errors (timeouts, 429, 5xx) are retried with backoff (`server.max_retries`,
`retry_backoff_s`); after `max_consecutive_errors` failures in a row the benchmark stops so it
can be resumed later instead of collecting errors.

Progress is printed every 50 questions:

```
  [ 50/285]  17%  7.3 q/s  ETA 00:32  errors: 0
```

At the end a summary is printed (including `n_items`, `n_expected` and `n_api_errors` per benchmark) and a log is written to the run directory:

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

Available names: `medqa`, `rar`, `radbench`, `vqa_med_2019`, `radimagenet_vqa`, `radiorag`, `label_extraction_mamma`, `label_extraction_arm`, `label_extraction` (needs its own data file, so `all` fails without it)

### Limiting samples (for quick tests)

```yaml
benchmark_settings:
  limit_samples: 100   # null = full dataset
```

---

## Output Files

| File                                                 | Contents                                           |
| ---------------------------------------------------- | -------------------------------------------------- |
| `results/run_<timestamp>/{benchmark}_results.csv`  | Raw model answers, one row per question            |
| `results/run_<timestamp>/{benchmark}_report.jsonl` | Full evaluation report with all metrics            |
| `results/run_<timestamp>/{benchmark}_judge_cache.csv` | Raw LLM-judge verdicts (reused on re-evaluation) |
| `results/run_<timestamp>/run_<timestamp>.log`      | Full run log including verbose per-question output |
| `results/run_<timestamp>/run_info_<timestamp>.json`| Config snapshot (without API keys), benchmarks, git commit |
| `results/run_<timestamp>/benchmark_results_<model>.png` | Bar chart of all benchmark scores             |

Results of the label-extraction benchmarks contain model output derived from patient reports;
`results/` is git-ignored and must not be shared.

---

## Re-running Evaluation

Evaluation runs automatically after each benchmark. To re-evaluate existing results manually:

```bash
# MCQ benchmarks (MedQA, RaR, RadioRAG)
python evaluate.py results/run_<timestamp>/medqa_results.csv --type mcq

# VQA benchmarks (RadBench, VQA-Med-2019, RadImageNet-VQA)
python evaluate.py results/run_<timestamp>/radbench_results.csv --type vqa

# VQA + LLM-as-a-Judge for open-ended questions
python evaluate.py results/run_<timestamp>/radimagenet_vqa_results.csv --type vqa --judge

# Label Extraction
python evaluate.py results/run_<timestamp>/label_extraction_results.csv --type extraction

# Mamma-MRT / Arm X-ray extraction
python evaluate.py results/run_<timestamp>/label_extraction_mamma_results.csv --type mamma_extraction
python evaluate.py results/run_<timestamp>/label_extraction_arm_results.csv --type arm_extraction
```

---

## Benchmark Interpretation

| Benchmark        | What it measures                                          | Metric                      | Random baseline | Notes                                        |
| ---------------- | --------------------------------------------------------- | --------------------------- | :-------------: | -------------------------------------------- |
| MedQA            | USMLE Step 1–3 clinical reasoning (4-choice MCQ)         | Accuracy                    |       25%       | General medicine knowledge                   |
| RaR              | Radiology board-style MCQ (5-choice)                      | Accuracy                    |       20%       | Domain-specific radiology reasoning          |
| RadioRAG         | Radiology factual QA (4-choice MCQ)                       | Accuracy                    |       25%       | Originally open-ended; converted to MCQ      |
| RadBench MCQ     | X-ray clinical questions with image (2–12 options)        | Accuracy                    |  1/#options     | Radiopaedia cases only: 285 questions (63 MCQ, 148 yes/no, 74 open); 212 MedPix questions dropped (images no longer available) |
| RadBench Yes/No  | Binary image questions (e.g. "Is there a pneumothorax?")  | Accuracy                    |       50%       | Requires VLM                                 |
| RadBench Open    | Free-text description of X-ray findings                   | Exact / WBSS / LLM-Judge    |       —         | LLM-Judge is the primary metric              |
| VQA-Med-2019     | Medical image VQA: modality, organ, plane, abnormality    | Exact / WBSS / LLM-Judge    |       —         | 500 items (ImageCLEF 2019); official metric: exact-match accuracy |
| RadImageNet-VQA  | CT/MRI: MCQ (2000), yes/no (5000), open pathology (2000)  | Accuracy / LLM-Judge        |   25% / 50%     | 9K-item benchmark split; 1K images           |
| Mamma-MRT        | Menopause, BI-RADS and BPE ("ACR") per side, lesion types per side | Accuracy, Macro-F1, Micro-F1 | — | 302 exams; fields with empty GT are not scored |
| Arm X-ray        | 18–28 binary findings per region + supporting citation   | Micro/Macro-F1, Sens., Spec. | — | 1371 test reports (clavicle 233, elbow 745, thumb 393) |
| Label Extraction | Entity extraction from radiology reports (NER)            | Micro F1                    |       —       | Higher = more complete entity set            |

---

## Metrics

### Accuracy

Rule-based letter extraction for MCQ (only capital option letters count, so "I think…" is not read as option I). Yes/No: the first word of the answer must be the expected "yes"/"no" (as in RadImageNet-VQA). Failed API calls count as wrong and are reported as `n_api_errors`. Random baseline: 1/#options, 50% for Yes/No.

### Exact match (open questions)

Normalised string equality (lowercase, punctuation removed) between answer and reference — the official VQA-Med-2019 accuracy.

### WBSS — Word-Based Semantic Similarity

Measures semantic similarity between model answer and reference answer using Wu-Palmer similarity from WordNet. Gives partial credit for synonyms and paraphrases.

| WBSS    | Interpretation                            |
| ------- | ----------------------------------------- |
| < 30%   | Poor                                      |
| 30–50% | Moderate                                  |
| 50–70% | Good — correct domain, different wording |
| > 70%   | Strong                                    |

### Micro F1

TP/FP/FN aggregated across all items before computing precision/recall (RadGraph protocol). Used for Label Extraction, Mamma lesions and Arm labels.

### Mamma-MRT extraction

- Fields (menopause, BI-RADS left/right, BPE left/right): Accuracy and Macro-F1 over the GT classes. "ACR" in the GT is background parenchymal enhancement (1–4), not breast density. BI-RADS 6 is mapped to 5 (`birads6_handling`); GT has no 6. If the GT field is empty, the model value is not scored (`gt_empty_ext_present: ignore`).
- Lesions per side: main metric = set of lesion types (Micro-F1, exact match); a count-based view (one entry per GT lesion row) is reported as `*_count_*`. Sides without GT lesions are not scored by default. Lesion names are mapped via `config/mamma_normalization.yaml`; mixed findings such as "ca/dcis" count as one lesion of the higher-grade type.

### Arm X-ray extraction

Labels are scored per region (`clavicle | Fracture` ≠ `elbow | Fracture`). Macro-F1 averages only labels with at least one positive in GT or prediction. Citation match = share of citations (for findings marked present) that occur verbatim in the report (case/whitespace-insensitive, ≥ 4 characters); it is checked while running, because the report text is not stored in the results.

### Confidence intervals

95% bootstrap CIs (1000 resamples over reports, seed 42) for Mamma accuracies and lesion Micro-F1 and for Arm Micro-F1; Micro-F1 is recomputed on every resample.

### LLM-as-a-Judge

A second LLM (configured in `config.yaml` under `judge:`) scores each open-ended answer as 0 or 1; it runs automatically whenever a `judge:` block exists (`evaluate.py --judge` for manual re-evaluation). Only a reply consisting of exactly `0` or `1` counts; other replies are reported as `n_judge_unparsed` and excluded from the judge accuracy. For reasoning models disable thinking via `extra_body` (see `config.default.yaml`), otherwise the answer can be empty. Verdicts are cached in `{benchmark}_judge_cache.csv`.

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
