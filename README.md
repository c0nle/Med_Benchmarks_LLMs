# Med_Benchmarks_LLMs

A modular, resumable evaluation framework for benchmarking LLMs and Vision-Language Models (VLMs) on medical AI tasks.
Models are queried through any OpenAI-compatible endpoint (LiteLLM / vLLM), so locally hosted open-weight models can be
evaluated without data leaving the institution.

---

## Benchmarks

| Benchmark (config name)                  | Type                                         | Requires VLM | Metric(s)                                                       |
| ---------------------------------------- | -------------------------------------------- | :----------: | --------------------------------------------------------------- |
| MedQA (`medqa`)                          | Text MCQ (USMLE, 4 options)                  |      —       | Accuracy                                                        |
| RaR (`rar`)                              | Text MCQ (radiology, 5 options)              |      —       | Accuracy                                                        |
| RadioRAG (`radiorag`)                    | Text MCQ variant (radiology, 4 options)      |      —       | Accuracy                                                        |
| RadBench (`radbench`)                    | X-ray image VQA (MCQ / yes-no / open)        |      ✅      | MCQ Acc. / Yes-No Acc. / Open: LLM-Judge (+ Exact, WBSS)        |
| VQA-Med-2019 (`vqa_med_2019`)            | Medical image VQA (open)                     |      ✅      | LLM-Judge, Exact match (+ WBSS); per category                   |
| RadImageNet-VQA (`radimagenet_vqa`)      | CT/MRI image VQA (MCQ / yes-no / open)       |      ✅      | MCQ Acc. / Yes-No Acc. / Open: LLM-Judge; per content type      |
| Mamma-MRT extraction (`label_extraction_mamma`) | Structured extraction, German breast MRI reports | — | Accuracy (+CI), Macro-F1, coverage per field; exam-level BPE accuracy; lesion-type Micro-P/R/F1 (+CIs) / Exact; sensitivity analyses |
| Arm X-ray extraction (`label_extraction_arm`)   | Binary labels + citation, German arm X-ray reports | — | Micro-/Macro-F1 and MCC (overall and per region, 95% CIs), Sens./Spec. per label, verbatim-citation rate |
| Label Extraction (`label_extraction`)    | NER (entity strings)                         |      —       | Entity-string Micro F1 (needs `data/extraction.parquet`, not included) |

> VLM benchmarks always send the image(s), so the model must accept image input.
> Multi-image RadBench questions send all images of the case; a question is never sent with fewer images.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt          # exact reference environment: requirements.lock
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
python -m pytest tests -q                # unit tests (synthetic data only)
```

### 2. Create your config

```bash
cp config.default.yaml config.yaml
```

Edit `config.yaml`:

```yaml
server:
  url: "https://your-server:4000/v1"   # LiteLLM / vLLM / OpenAI-compatible endpoint
  model_name: "google/gemma-4-31B-it"
  api_key: "sk-your-key-here"          # or leave empty and set api_key_env
  seed: 42
  timeout_s: 300                       # long JSON answers (label extraction) need > 60 s
  # extra_body:                        # e.g. disable thinking of reasoning models
  #   chat_template_kwargs:
  #     enable_thinking: false

judge:                                 # LLM-as-a-Judge for open-ended answers
  url: "https://your-server:4000/v1"
  model_name: "your-judge-model"
  temperature: 0
  max_tokens: 512

benchmark: medqa                        # see below for all options

benchmark_settings:
  limit_samples: 20                     # start small for a first test; null = full dataset
  temperature: 0.0
  max_tokens: 256
  concurrency: 4                        # parallel requests per benchmark
```

`config.yaml` is git-ignored. For several models, keep one config per model in `configs_local/` (also git-ignored)
and pass it with `--config`.

### 3. Download datasets

Most benchmarks load automatically. A few require manual download:

| Benchmark        | Source                                                                                               | Place at                                     |
| ---------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| MedQA            | Auto (HuggingFace) or [openlifescienceai/medqa](https://huggingface.co/datasets/openlifescienceai/medqa) | `data/medqa-test.parquet`                  |
| VQA-Med-2019     | Auto (HuggingFace) or [simwit/vqa-med-2019](https://huggingface.co/datasets/simwit/vqa-med-2019)         | `data/vqa_med_2019.parquet`                |
| RadImageNet-VQA  | [raidium/RadImageNet-VQA](https://huggingface.co/datasets/raidium/RadImageNet-VQA)                       | `data/radimagenet_vqa_benchmark.parquet`   |
| RadBench         | [harrison-ai/radbench](https://github.com/harrison-ai/radbench); images: `python scripts/download_radbench_images.py` | `data/radbench.csv`, `data/radbench_images_v2/` |
| RaR              | Contact authors via [paper](https://www.nature.com/articles/s41746-025-02250-5)                          | `data/RaR_dataset_WithAnswer.csv`          |
| RadioRAG         | Contact authors via [GitHub](https://github.com/tayebiarasteh/RadioRAG)                                  | `data/RadioRAG_WithOptions_WithAnswer.csv` |
| Mamma-MRT        | Not public (local patient data)                                                                      | `data/label_extraction/label_extraction_gt.xlsx` (ground truth) + `hiwi_gt_ergaenzung.xlsx` (report texts; its `*_GT` columns are identical to the ground truth and only used as a consistency check) |
| Arm X-ray        | Not public (local patient data; Kreutzer et al., Eur Radiol 2025)                                    | `data/label_extraction/Label_Extraction_Kilian/` |
| Label Extraction | Your own radiology NER dataset                                                                       | `data/extraction.parquet`                  |

RadBench images are stored as `sha1(url)[:16]` + extension with a `manifest.csv` (url, file, sha256). A missing image
stops the run; set `task_settings.radbench.missing_images: skip_question` to skip such questions instead.

### 4. Run

```bash
python main.py
```

---

## Running Benchmarks

```bash
python main.py                                    # what is set in config.yaml
python main.py --limit 5                          # quick test, 5 items per benchmark
python main.py --benchmark medqa,radbench         # select benchmarks
python main.py --config configs_local/qwen.yaml   # another config (e.g. one per model)
python main.py --model google/gemma-4-31B-it      # override server.model_name
python main.py --run-dir results/run_<timestamp>  # resume: finished items are skipped,
                                                  # failed requests and missing judge verdicts are redone
```

On the HPC cluster submit `run.sh` from the project directory (arguments are passed to `main.py`):

```bash
sbatch run.sh --config configs_local/<model>.yaml --run-dir results/study/<model>
sbatch run.sh --config configs_local/<model>.yaml --run-dir results/study/<model>   # resume after a timeout
```

**Startup checks.** Before any benchmark runs, `GET /v1/models` must list the model and the judge, otherwise the run
stops (`--skip-health-check` disables this). 401/403/404 stop the run immediately.

**Retries.** Transient errors (timeouts, 429, 5xx) are retried with backoff (`max_retries` 2, `retry_backoff_s` 5);
after `max_consecutive_errors` (10) failures in a row the benchmark stops, the remaining ones are skipped, and the run
is marked INCOMPLETE so it can be resumed.

**Concurrency and locks.** `benchmark_settings.concurrency` (default 4) is the number of parallel requests per
benchmark; only one thread writes the results CSV. Two jobs may share a `--run-dir` only for different benchmarks;
a second job on the same benchmark exits with "already running" (lock file `.<benchmark>.lock`).

**Resume safety.** `fingerprint.json` stores model, judge, sampling settings and the system-prompt hash; resuming
with a different model or judge is refused (`--force-resume` overrides). Duplicate result rows are reduced to the
latest answer per id; answers for items outside the current `--limit` are kept but not evaluated.

**Exit codes.** 0 = all benchmarks complete; 1 = a benchmark failed or is incomplete (including API errors);
2 = lock or startup configuration error.

At the end a summary is printed (with `n_items`, `n_expected`, `n_api_errors`, headline metrics and 95% CIs; incomplete
benchmarks are flagged `INCOMPLETE`) and a chart is written to the run directory.

### Selecting benchmarks

```yaml
benchmark: medqa                                  # single
benchmark: [medqa, radbench, vqa_med_2019]        # several
benchmark: all                                    # all registered
```

Available names: `medqa`, `rar`, `radbench`, `vqa_med_2019`, `radimagenet_vqa`, `radiorag`, `label_extraction_mamma`, `label_extraction_arm`, `label_extraction` (needs its own data file, so `all` fails without it)

`limit_samples` takes the first N items of each benchmark (deterministic, not stratified).

### Comparing models

```bash
python scripts/compare_models.py results/study/model_a results/study/model_b [--labels "A,B"] [--out-dir DIR]
```

Writes `comparison.csv` (benchmark, metric, model, value, CI, n, complete), `comparison.png` and `mcnemar.csv`
(paired exact McNemar test of each run against the first one on shared items).

---

## Output Files

| File                                                 | Contents                                           |
| ---------------------------------------------------- | -------------------------------------------------- |
| `{benchmark}_results.csv`                            | Raw model answers, one row per item (incl. category, cluster_id, finish_reason, completion_tokens) |
| `{benchmark}_report.jsonl`                           | Evaluation report: metric rows (with `ci_lo`/`ci_hi`) and item rows |
| `{benchmark}_status.json`                            | n_items, n_expected, n_api_errors, complete, stop_reason, finished_at |
| `{benchmark}_judge_cache_{judge-model}.csv`          | Raw LLM-judge verdicts, keyed by id + answer/reference hash + judge model + prompt version |
| `fingerprint.json`                                   | Settings checked on resume                         |
| `run_<ts>_<job>.log`                                 | Full run log (stdout and stderr) including verbose per-item output |
| `run_info_<ts>_<job>.json`                           | Config (without API keys), CLI args, host, SLURM job, package versions, git status + diff hash, served and server-reported models, per-benchmark status |
| `benchmark_results_<model>.png`                      | One panel per family; headline metrics with 95% CI, n per bar, chance levels; incomplete benchmarks hatched |

Results, reports and logs of the label-extraction benchmarks contain model output derived from patient reports;
`results/` is git-ignored and must not be shared.

---

## Re-running Evaluation

Evaluation runs automatically after each benchmark. To re-evaluate existing results manually:

```bash
python evaluate.py results/<run>/medqa_results.csv --type mcq
python evaluate.py results/<run>/radbench_results.csv --type vqa
python evaluate.py results/<run>/radimagenet_vqa_results.csv --type vqa --judge            # judge from config
python evaluate.py results/<run>/radimagenet_vqa_results.csv --type vqa --judge-model NAME # second judge (agreement)
python evaluate.py results/<run>/label_extraction_mamma_results.csv --type mamma_extraction --config configs_local/<model>.yaml
python evaluate.py results/<run>/label_extraction_arm_results.csv --type arm_extraction
```

`--config` (default `config.yaml`, else `config.default.yaml`) supplies the judge and the `task_settings`.

---

## Prompts

- MCQ: question + options, "Reply with only the correct letter".
- Yes/No: "Reply with only 'Yes' or 'No'".
- Open-ended VQA: "Answer the question with a single word or a short phrase." (before 2026-10 the prompt asked for
  "key medical terms only"; results from older runs are not comparable).
- Mamma-MRT: German system and user prompt with a JSON schema. BI-RADS is asked as 2–5 by imaging finding, also
  for an already histologically proven carcinoma (the annotation never uses 6; before 2026-10-07 the prompt offered
  6 = proven malignancy and models used it in staging reports). Arm X-ray: JSON with one entry per label and a
  verbatim citation; the datasets contain no label definitions, so the prompt lists label names only.
- Default system prompt for public benchmarks: "You are a medical expert in diagnostic imaging. Answer concisely and
  in English."

---

## Benchmark Interpretation

| Benchmark        | What it measures                                          | Metric                      | Random baseline | Notes                                        |
| ---------------- | --------------------------------------------------------- | --------------------------- | :-------------: | -------------------------------------------- |
| MedQA            | USMLE Step 1–3 clinical reasoning (4-choice MCQ)         | Accuracy                    |       25%       | General medicine knowledge                   |
| RaR              | Radiology board-style MCQ (5-choice)                      | Accuracy                    |       20%       | Small (n=65) and near ceiling; answer key skewed (C = 27/65) |
| RadioRAG         | Radiology factual QA (4-choice MCQ)                       | Accuracy                    |       25%       | Originally open-ended; converted to MCQ      |
| RadBench         | X-ray questions with image(s)                             | Accuracy / LLM-Judge        | 1/#options; 50% yes/no | Radiopaedia cases only: 283 questions (63 MCQ, 147 yes/no, 73 open) on 49 cases; 212 MedPix questions dropped (images unavailable); 2 questions of case 77654 dropped (image reference `52662257` is not a URL) |
| VQA-Med-2019     | Medical image VQA: modality, plane, organ, abnormality    | LLM-Judge / Exact match     |       —         | 500 items (125 per category); official metric: exact-match accuracy |
| RadImageNet-VQA  | CT/MRI: MCQ (2000), yes/no (5000), open (2000)            | Accuracy / LLM-Judge        |   25% / 50%     | 9K items on 1K images; per content type (anatomy / pathology) |
| Mamma-MRT        | Menopause, BI-RADS and BPE ("ACR") per side, lesion types per side | Accuracy, Macro-F1, Micro-F1 | majority class | 302 exams; see definitions below |
| Arm X-ray        | 18–28 binary findings per region + supporting citation   | Micro/Macro-F1, MCC          | all-negative accuracy | 1371 test reports (clavicle 233, elbow 745, thumb 393) |
| Label Extraction | Entity strings from radiology reports                     | Micro F1                    |       —       | Not the RadGraph span protocol               |

---

## Metrics

### Accuracy

Rule-based letter extraction for MCQ: the last explicit answer statement counts ("answer is X", "Answer: X",
"Correct Letter: X", then **X**); two letters asserted together or several different letters → unparsed (wrong);
only the question's option letters are accepted. Yes/No: the first word of the answer must be the expected "yes"/"no".
Failed API calls count as wrong and are reported as `n_api_errors`.

### Exact match (open questions)

Normalised string equality (lowercase, punctuation removed) against any accepted reference (VQA-Med-2019 lists
several for 32 questions) — the official VQA-Med-2019 accuracy.

### WBSS — Word-Based Semantic Similarity

Secondary metric from VQA-Med 2018 (Wu-Palmer similarity over WordNet; re-implementation, not the official scorer).
It is reported together with `wbss_shuffled_baseline_pct` (references permuted, seed 42): unrelated answers already
score ≈ 37–46%, so WBSS is not used as a headline metric.

### LLM-as-a-Judge

A separate model (configured under `judge:`) scores each open-ended answer as 0 or 1 against all reference answers,
with a short rubric. Unparsed replies and judge errors count as wrong (`n_judge_unparsed`, `n_judge_errors`).
For reasoning models disable thinking via `extra_body`. A second judge can be run with `--judge-model` for an
agreement analysis; verdicts are cached per judge model.

### Micro F1

TP/FP/FN aggregated across all items before computing precision/recall. Used for Label Extraction (entity strings),
Mamma lesions and Arm labels.

### Mamma-MRT extraction

- Fields (menopause, BI-RADS left/right, BPE left/right): Accuracy (95% CI), Macro-F1 over the GT classes and
  coverage (share of scored items where the model gave a value; a missing value is wrong for accuracy but only an FN
  for macro-F1, so macro-F1 can exceed accuracy). "ACR" in the GT is background parenchymal enhancement (1–4), not
  breast density. BPE is usually one value per exam, so the left/right rows are not independent;
  `acr_exam_accuracy_pct` scores one decision per exam (exams whose GT differs between sides are excluded and counted).
- Primary definition: BI-RADS 6 is mapped to 5 (`birads6_handling`; the GT has no 6 and the prompt asks for 2–5, so this only catches stray answers) and fields or lesion sides with
  empty GT are not scored (`gt_empty_ext_present: ignore`). Sensitivity analyses with the other option of each setting
  are always reported, using the same metric name with a tag before `_pct` (`birads_li_accuracy_birads6keep_pct`,
  `menopause_accuracy_gtemptyfp_pct`, `lesions_li_micro_f1_gtemptyfp_pct`; JSONL rows carry a `variant` field),
  together with the number of BI-RADS-6 answers and of ignored model values.
- Lesions per side: main metric = set of lesion types (micro precision/recall/F1 with CIs, exact match); a
  count-based view (one entry per GT lesion row) is reported as `*_count_*`. Lesion names are mapped via
  `config/mamma_normalization.yaml`; mixed findings such as "ca/dcis" count as one lesion of the higher-grade type;
  lymph-node metastases map to "sonstige Läsion" (not to the benign "Lymphknoten").

### Arm X-ray extraction

Labels are scored per region (`clavicle | Fracture` ≠ `elbow | Fracture`). Main metrics: micro-/macro-F1 and MCC,
overall and per region. Macro-F1 averages only labels with at least one positive in GT or prediction. Accuracy is
secondary because most labels are absent; compare `all_negative_baseline_accuracy_pct`. Unparseable answers are not
scored (`n_parse_error`); labels missing from an answer are counted as missing (`n_missing_labels`), not as negative.
`verbatim_citation_rate_pct` is the share of citations (for findings marked present) that occur verbatim in the
report (case/whitespace-insensitive, ≥ 4 characters). It is split into true-positive and false-positive calls,
because a verbatim quote is not evidence that the finding is correct. It is checked at run time, because the report
text is not stored. `citation_match_pct` is a deprecated alias.

### Confidence intervals

- Public benchmarks: Wilson 95% CI; where items share an image/case (RadBench case, RadImageNet image, VQA-Med image)
  a cluster bootstrap (1000 resamples, seed 42, percentile) is the reported CI and Wilson is kept as
  `ci_lo_wilson`/`ci_hi_wilson`. Per-category rows use subsets like `open:category=plane`.
- Mamma / Arm: 95% bootstrap CIs (1000 resamples over reports, seed 42) for Mamma accuracies and lesion micro
  precision/recall/F1, and for Arm micro-/macro-F1 and MCC overall and per region. Scores are recomputed on every
  resample.

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
- **Arm X-ray**: Kreutzer et al. (2025). European Radiology. https://doi.org/10.1007/s00330-025-12102-1
- **RadGraph** (background for Label Extraction): Jain et al. (2021). NeurIPS 2021. https://physionet.org/content/radgraph/

**Evaluation Methodology**

- **WBSS**: Hasan et al. (2018). *Overview of ImageCLEF 2018 Medical Domain Visual Question Answering Task.* CLEF 2018.
- **LLM-as-a-Judge**: Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023. https://arxiv.org/abs/2306.05685
