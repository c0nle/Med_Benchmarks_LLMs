import argparse
import re
import string
from typing import Optional
import json
import warnings

import pandas as pd


CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def extract_choice(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    match = CHOICE_RE.search(text)
    if match:
        return match.group(1).upper()
    return None


def score_results(df: pd.DataFrame) -> pd.DataFrame:
    if "correct_answer" not in df.columns or "model_answer" not in df.columns:
        raise ValueError("CSV must contain columns: correct_answer, model_answer")

    df = df.copy()
    df["correct_answer_norm"] = df["correct_answer"].map(extract_choice)
    df["model_answer_norm"] = df["model_answer"].map(extract_choice)
    df["is_correct"] = df["correct_answer_norm"] == df["model_answer_norm"]
    return df


def compute_reports(scored_df: pd.DataFrame):
    total = len(scored_df)
    parsed_model = int(scored_df["model_answer_norm"].notna().sum())
    parsed_correct = int(scored_df["correct_answer_norm"].notna().sum())
    accuracy = float(scored_df["is_correct"].mean() * 100) if total else 0.0

    dist = (
        scored_df["model_answer_norm"]
        .fillna("UNPARSED")
        .value_counts(dropna=False)
        .rename_axis("answer")
        .reset_index(name="count")
    )
    dist["pct"] = ((dist["count"] / total * 100).round(2) if total else 0.0)

    conf = pd.crosstab(
        scored_df["correct_answer_norm"].fillna("UNPARSED"),
        scored_df["model_answer_norm"].fillna("UNPARSED"),
        dropna=False,
    )

    metrics = pd.DataFrame(
        [
            ("rows", total),
            ("parsed_correct_answer", parsed_correct),
            ("parsed_model_answer", parsed_model),
            ("accuracy_pct", round(accuracy, 2)),
        ],
        columns=["metric", "value"],
    )

    return metrics, dist, conf


def _jsonable(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return value


def write_report_jsonl(
    results_csv_path: str = "results/benchmark_results.csv",
    out_path: str = "results/benchmark_report.jsonl",
) -> dict:
    """
    Writes a single JSONL file containing:
    - metrics rows (type=metric)
    - answer distribution rows (type=answer_distribution)
    - confusion matrix rows (type=confusion, only non-zero cells)
    - per-item scored rows (type=item)
    """
    df = pd.read_csv(results_csv_path)
    scored = score_results(df)
    metrics, dist, conf = compute_reports(scored)

    accuracy_row = metrics.loc[metrics["metric"] == "accuracy_pct", "value"]
    accuracy_pct = float(accuracy_row.iloc[0]) if not accuracy_row.empty else 0.0

    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in metrics.iterrows():
            obj = {"type": "metric", "metric": row["metric"], "value": row["value"]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        for _, row in dist.iterrows():
            obj = {
                "type": "answer_distribution",
                "answer": _jsonable(row.get("answer")),
                "count": _jsonable(row.get("count")),
                "pct": _jsonable(row.get("pct")),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        for correct in conf.index:
            for model in conf.columns:
                count = int(conf.loc[correct, model])
                if count == 0:
                    continue
                obj = {
                    "type": "confusion",
                    "correct_answer": _jsonable(correct),
                    "model_answer": _jsonable(model),
                    "count": count,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        for _, row in scored.iterrows():
            obj = {"type": "item"}
            for col, val in row.to_dict().items():
                obj[col] = _jsonable(val)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return {"accuracy_pct": accuracy_pct, "path": out_path}


def print_terminal_report(results_csv_path: str = "results/benchmark_results.csv") -> None:
    df = pd.read_csv(results_csv_path)
    scored = score_results(df)
    metrics, _, _ = compute_reports(scored)

    accuracy_row = metrics.loc[metrics["metric"] == "accuracy_pct", "value"]
    accuracy_pct = float(accuracy_row.iloc[0]) if not accuracy_row.empty else 0.0
    rows = int(metrics.loc[metrics["metric"] == "rows", "value"].iloc[0]) if not metrics.empty else len(df)

    print(f"  Accuracy: {accuracy_pct:.2f}%  ({rows} questions)")


# ===========================================================================
# VQA Evaluation
# ===========================================================================

def _normalise_text(text: str, stem: bool = False) -> str:
    """
    Basic normalisation: lowercase, hyphen/slash → space, strip punctuation.
    Used for exact-match, token-F1, and WBSS.
    """
    text = str(text).lower().strip()
    text = text.replace("-", " ").replace("/", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _paper_tokenise(text: str) -> list:
    """
    Exact preprocessing from the VQA-Med-2019/2020 official evaluator
    (Ben Abacha et al., ImageCLEF 2019; source: Evaluator-VQA-Med-2020.py):
      1. Lowercase
      2. Strip string.punctuation
      3. word_tokenize (NLTK)
      4. Remove English stopwords           ← 'no' IS a stopword → removed!
      5. Snowball stemming

    Note: because 'no' is an NLTK English stopword, answers of 'no' become []
    and receive BLEU=0 even when correct. This is a quirk of the official script.
    """
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    try:
        from nltk.corpus import stopwords as _sw
        sw = set(_sw.words("english"))
        tokens = [t for t in tokens if t not in sw]
    except Exception:
        pass
    try:
        from nltk.stem import SnowballStemmer
        stemmer = SnowballStemmer("english")
        tokens = [stemmer.stem(t) for t in tokens]
    except Exception:
        pass
    return tokens


def _token_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 between prediction and reference strings.
    Used for open-ended VQA scoring (SQuAD-style).
    """
    pred_tokens = _normalise_text(prediction).split()
    ref_tokens = _normalise_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len([t for t in pred_tokens if t in common]) / len(pred_tokens)
    recall = len([t for t in ref_tokens if t in common]) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, reference: str) -> bool:
    return _normalise_text(prediction) == _normalise_text(reference)


def score_vqa_mcq(df: pd.DataFrame) -> pd.DataFrame:
    """Score MCQ rows in a VQA results CSV (columns: reference_answer, model_answer)."""
    df = df.copy()
    df["correct_answer_norm"] = df["reference_answer"].map(extract_choice)
    df["model_answer_norm"] = df["model_answer"].map(extract_choice)
    df["is_correct"] = df["correct_answer_norm"] == df["model_answer_norm"]
    return df


def _vqa_med_bleu(prediction: str, reference: str) -> float:
    """
    Per-item BLEU exactly as in the VQA-Med-2019/2020 official evaluator
    (Ben Abacha et al., ImageCLEF 2019; source: Evaluator-VQA-Med-2020.py):
      - _paper_tokenise(): lowercase → strip punctuation → word_tokenize
                           → remove English stopwords → Snowball stem
      - sentence_bleu with SmoothingFunction().method0
      - Default weights (0.25, 0.25, 0.25, 0.25) = BLEU-4

    Known quirk: 'no' is an NLTK stopword → tokenises to [] → always BLEU=0,
    even when correct. This matches the official evaluator behaviour exactly.
    Top systems 2019: BLEU ~64%, Accuracy ~62%.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        pred_tokens = _paper_tokenise(prediction)
        ref_tokens  = _paper_tokenise(reference)
        # Official evaluator: if BOTH are empty after preprocessing (e.g. "no" vs "no",
        # since "no" is an English stopword), assign score = 1.0 (exact match).
        if len(ref_tokens) == 0 and len(pred_tokens) == 0:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sentence_bleu(
                [ref_tokens], pred_tokens,
                smoothing_function=SmoothingFunction().method0,
            )
    except ImportError:
        return _token_f1(prediction, reference)


def _corpus_bleu(predictions: list, references: list) -> float:
    """Average per-item VQA-Med BLEU across all pairs (paper methodology)."""
    if not predictions:
        return 0.0
    scores = [_vqa_med_bleu(p, r) for p, r in zip(predictions, references)]
    return sum(scores) / len(scores)


def _sentence_bleu(prediction: str, reference: str) -> float:
    """
    Sentence-level BLEU for per-item storage (not the paper's primary metric).
    Used only to populate the per-row 'bleu' column; aggregate via _corpus_bleu.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        pred_tokens = _normalise_text(prediction).split()
        ref_tokens = _normalise_text(reference).split()
        if not pred_tokens or not ref_tokens:
            return 0.0
        sf = SmoothingFunction().method1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=sf)
    except ImportError:
        return _token_f1(prediction, reference)


def _wbss(prediction: str, reference: str) -> float:
    """
    Word-Based Semantic Similarity (WBSS) via Wu-Palmer similarity on WordNet.
    Used in VQA-Med-2019 alongside BLEU (Ben Abacha et al., ImageCLEF 2019).
    Requires: nltk + nltk.download('wordnet') + nltk.download('omw-1.4')
    Returns 0.0 if WordNet is unavailable.
    """
    try:
        from nltk.corpus import wordnet as wn
        pred_tokens = _normalise_text(prediction).split()
        ref_tokens = _normalise_text(reference).split()
        if not pred_tokens or not ref_tokens:
            return 0.0

        def best_wup(word, candidates):
            syns_w = wn.synsets(word)
            if not syns_w:
                return 0.0
            best = 0.0
            for cand in candidates:
                for sc in wn.synsets(cand):
                    for sw in syns_w:
                        sim = sw.wup_similarity(sc)
                        if sim and sim > best:
                            best = sim
            return best

        p2r = sum(best_wup(w, ref_tokens) for w in pred_tokens) / len(pred_tokens)
        r2p = sum(best_wup(w, pred_tokens) for w in ref_tokens) / len(ref_tokens)
        if p2r + r2p == 0:
            return 0.0
        return 2 * p2r * r2p / (p2r + r2p)
    except Exception:
        return 0.0


def score_vqa_open(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score open-ended VQA rows.

    Primary metric: WBSS (Wu-Palmer semantic similarity via WordNet).
    LLM-as-a-Judge is done separately via evaluate_vqa_with_judge().
    """
    df = df.copy()
    df["wbss"] = df.apply(
        lambda r: _wbss(str(r["model_answer"]), str(r["reference_answer"])), axis=1
    )
    return df


def evaluate_vqa_with_judge(
    df: pd.DataFrame,
    client,
) -> pd.DataFrame:
    """
    LLM-as-a-Judge evaluation for open-ended VQA rows.

    Implements the binary correct/incorrect rubric used by RadImageNet-VQA
    (Butsanets et al., 2025, following Zheng et al., 2023):
    The judge receives the question, the ground-truth answer, and the model
    prediction and returns 1 (correct) or 0 (incorrect).

    Returns a copy of df with an added 'judge_correct' column (int 0|1 or NaN).
    """
    df = df.copy()
    scores = []
    for _, row in df.iterrows():
        prompt = (
            "You are a medical expert judge evaluating a model's answer to a radiology question.\n\n"
            f"Question: {row['question']}\n"
            f"Ground-truth answer: {row['reference_answer']}\n"
            f"Model answer: {row['model_answer']}\n\n"
            "Is the model answer medically correct and equivalent in meaning to the ground-truth answer?\n"
            "Reply with exactly '1' (correct) or '0' (incorrect). No other text."
        )
        raw = client.ask_question(prompt)
        try:
            match = re.search(r"\b([01])\b", str(raw))
            score = int(match.group(1)) if match else None
        except Exception:
            score = None
        scores.append(score)
    df["judge_correct"] = scores
    return df


def write_vqa_report_jsonl(
    results_csv_path: str,
    out_path: str,
    client=None,
    run_judge: bool = False,
) -> dict:
    """
    Evaluate a VQA results CSV and write a JSONL report.

    MCQ rows  → letter-accuracy (same logic as MCQ benchmarks).
    Open rows → exact match, token F1, and optionally LLM-as-a-Judge score.
    """
    df = pd.read_csv(results_csv_path)

    # Split by question type (column may be absent for pure-open datasets)
    q_type_col = "question_type" if "question_type" in df.columns else None
    if q_type_col:
        mcq_df = df[df[q_type_col] == "mcq"].copy()
        yes_no_df = df[df[q_type_col] == "yes_no"].copy()
        open_df = df[df[q_type_col] == "open"].copy()
    else:
        mcq_df = pd.DataFrame()
        yes_no_df = pd.DataFrame()
        open_df = df.copy()

    results: dict = {"path": out_path}

    with open(out_path, "w", encoding="utf-8") as f:
        # --- MCQ sub-results ---
        if not mcq_df.empty:
            # Rename for compatibility with score_results
            scored_mcq = score_vqa_mcq(mcq_df)
            accuracy = float(scored_mcq["is_correct"].mean() * 100)
            results["mcq_accuracy_pct"] = round(accuracy, 2)
            f.write(json.dumps({"type": "metric", "subset": "mcq", "metric": "accuracy_pct", "value": round(accuracy, 2)}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"type": "metric", "subset": "mcq", "metric": "rows", "value": len(scored_mcq)}, ensure_ascii=False) + "\n")
            for _, row in scored_mcq.iterrows():
                obj = {"type": "item", "subset": "mcq"}
                for col, val in row.to_dict().items():
                    obj[col] = _jsonable(val)
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # --- Yes/No (closed-ended) sub-results — RadImageNet-VQA binary task ---
        # Evaluated with exact-match accuracy (per paper)
        if not yes_no_df.empty:
            scored_yn = yes_no_df.copy()
            scored_yn["is_correct"] = scored_yn.apply(
                lambda r: _normalise_text(str(r["model_answer"])) == _normalise_text(str(r["reference_answer"])),
                axis=1,
            )
            yn_acc = float(scored_yn["is_correct"].mean() * 100)
            results["yes_no_accuracy_pct"] = round(yn_acc, 2)
            f.write(json.dumps({"type": "metric", "subset": "yes_no", "metric": "accuracy_pct", "value": round(yn_acc, 2), "note": "exact-match, RadImageNet-VQA closed-ended task"}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"type": "metric", "subset": "yes_no", "metric": "rows", "value": len(scored_yn)}, ensure_ascii=False) + "\n")
            for _, row in scored_yn.iterrows():
                obj = {"type": "item", "subset": "yes_no"}
                for col, val in row.to_dict().items():
                    obj[col] = _jsonable(val)
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # --- Open-ended sub-results ---
        if not open_df.empty:
            scored_open = score_vqa_open(open_df)
            if run_judge and client is not None:
                scored_open = evaluate_vqa_with_judge(scored_open, client)

            avg_wbss = float(scored_open["wbss"].mean() * 100)
            results["open_wbss_pct"] = round(avg_wbss, 2)

            f.write(json.dumps({"type": "metric", "subset": "open", "metric": "wbss_pct", "value": round(avg_wbss, 2), "note": "Wu-Palmer semantic similarity via WordNet"}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"type": "metric", "subset": "open", "metric": "rows", "value": len(scored_open)}, ensure_ascii=False) + "\n")

            # LLM-as-a-Judge (binary 0/1) — RadImageNet-VQA open-ended metric
            if "judge_correct" in scored_open.columns:
                valid = scored_open["judge_correct"].dropna()
                if not valid.empty:
                    judge_acc = float(valid.mean() * 100)
                    results["open_judge_accuracy_pct"] = round(judge_acc, 2)
                    f.write(json.dumps({"type": "metric", "subset": "open", "metric": "llm_judge_accuracy_pct", "value": round(judge_acc, 2), "note": "binary correct/incorrect, RadImageNet-VQA primary metric"}, ensure_ascii=False) + "\n")

            for _, row in scored_open.iterrows():
                obj = {"type": "item", "subset": "open"}
                for col, val in row.to_dict().items():
                    obj[col] = _jsonable(val)
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return results


def print_vqa_terminal_report(results_csv_path: str, report: dict = None) -> None:
    df = pd.read_csv(results_csv_path)
    q_type_col = "question_type" if "question_type" in df.columns else None

    if q_type_col:
        mcq_df = df[df[q_type_col] == "mcq"].copy()
        yes_no_df = df[df[q_type_col] == "yes_no"].copy()
        open_df = df[df[q_type_col] == "open"].copy()
    else:
        mcq_df = pd.DataFrame()
        yes_no_df = pd.DataFrame()
        open_df = df.copy()

    parts = []
    if not mcq_df.empty:
        scored = score_vqa_mcq(mcq_df)
        acc = float(scored["is_correct"].mean() * 100)
        parts.append(f"MCQ Accuracy: {acc:.2f}% ({len(scored)} questions)")

    if not yes_no_df.empty:
        yn_acc = float(yes_no_df.apply(
            lambda r: _normalise_text(str(r["model_answer"])) == _normalise_text(str(r["reference_answer"])), axis=1
        ).mean() * 100)
        parts.append(f"Yes/No Accuracy: {yn_acc:.2f}% ({len(yes_no_df)} questions)")

    if not open_df.empty:
        scored = score_vqa_open(open_df)
        wbss = float(scored["wbss"].mean() * 100)
        judge_str = ""
        if report and "open_judge_accuracy_pct" in report:
            judge_str = f"  LLM-Judge: {report['open_judge_accuracy_pct']:.2f}%"
        parts.append(f"Open WBSS: {wbss:.2f}% ({len(scored)} questions){judge_str}")

    for p in parts:
        print(f"  {p}")


# ===========================================================================
# Extraction Evaluation (Entity-F1)
# ===========================================================================

def _parse_entities(raw: str) -> set:
    """
    Parse a comma-separated entity string into a normalised set of tokens.
    Empty / error strings return an empty set.
    """
    if not raw or (isinstance(raw, float) and pd.isna(raw)):
        return set()
    raw = str(raw)
    if raw.startswith("Error:"):
        return set()
    entities = {_normalise_text(e) for e in raw.split(",") if e.strip()}
    return {e for e in entities if e}


def score_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-item TP/FP/FN for entity extraction.
    Micro F1 is computed globally in write_extraction_report_jsonl (RadGraph metric).
    Per-item columns added: tp, fp, fn (for aggregation).
    """
    df = df.copy()
    tps, fps, fns = [], [], []
    for _, row in df.iterrows():
        ref = _parse_entities(row.get("reference_entities", ""))
        pred = _parse_entities(row.get("model_entities", ""))
        tp = len(ref & pred)
        fp = len(pred - ref)
        fn = len(ref - pred)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
    df["tp"] = tps
    df["fp"] = fps
    df["fn"] = fns
    return df


def _micro_prf(scored_df: pd.DataFrame):
    """Compute global micro precision, recall, F1 from per-item TP/FP/FN."""
    total_tp = scored_df["tp"].sum()
    total_fp = scored_df["fp"].sum()
    total_fn = scored_df["fn"].sum()
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p * 100, 2), round(r * 100, 2), round(f * 100, 2)


def write_extraction_report_jsonl(
    results_csv_path: str,
    out_path: str,
) -> dict:
    """
    Evaluate label extraction results using Micro F1 (as in RadGraph, Jain et al. NeurIPS 2021).
    Micro F1 aggregates TP/FP/FN across all instances before computing precision/recall.
    """
    df = pd.read_csv(results_csv_path)
    scored = score_extraction(df)
    micro_p, micro_r, micro_f1 = _micro_prf(scored)

    with open(out_path, "w", encoding="utf-8") as f:
        for metric, value in [
            ("rows", len(scored)),
            ("micro_precision_pct", micro_p),
            ("micro_recall_pct", micro_r),
            ("micro_f1_pct", micro_f1),
        ]:
            f.write(json.dumps({"type": "metric", "metric": metric, "value": value}, ensure_ascii=False) + "\n")

        for _, row in scored.iterrows():
            obj = {"type": "item"}
            for col, val in row.to_dict().items():
                obj[col] = _jsonable(val)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return {
        "micro_precision_pct": micro_p,
        "micro_recall_pct": micro_r,
        "micro_f1_pct": micro_f1,
        "path": out_path,
    }


def print_extraction_terminal_report(results_csv_path: str) -> None:
    df = pd.read_csv(results_csv_path)
    scored = score_extraction(df)
    micro_p, micro_r, micro_f1 = _micro_prf(scored)
    print(f"  Micro F1: {micro_f1:.2f}%  P: {micro_p:.2f}%  R: {micro_r:.2f}%  ({len(scored)} questions)")


# ===========================================================================
# Open-ended QA Evaluation (RadioRAG)
# ===========================================================================

def write_open_qa_report_jsonl(
    results_csv_path: str,
    out_path: str,
    client=None,
    run_judge: bool = False,
) -> dict:
    """
    Evaluate open-ended QA results (RadioRAG).

    Automatic metrics: exact match, BLEU, token F1.
    Primary metric (RadioRAG paper): LLM-as-a-Judge binary accuracy.
    Run with run_judge=True (requires a live LLM in client).

    Tayebi Arasteh et al. 2024/2025 — human expert baseline: ~63% accuracy.
    """
    df = pd.read_csv(results_csv_path)
    scored = score_vqa_open(df)   # adds wbss

    if run_judge and client is not None:
        scored = evaluate_vqa_with_judge(scored, client)

    avg_wbss = float(scored["wbss"].mean() * 100)

    result = {
        "path": out_path,
        "wbss_pct": round(avg_wbss, 2),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "metric", "metric": "rows", "value": len(scored)}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"type": "metric", "metric": "wbss_pct", "value": round(avg_wbss, 2)}, ensure_ascii=False) + "\n")

        if "judge_correct" in scored.columns:
            valid = scored["judge_correct"].dropna()
            if not valid.empty:
                judge_acc = float(valid.mean() * 100)
                result["judge_accuracy_pct"] = round(judge_acc, 2)
                f.write(json.dumps({
                    "type": "metric",
                    "metric": "llm_judge_accuracy_pct",
                    "value": round(judge_acc, 2),
                    "note": "primary metric (RadioRAG paper); human baseline ~63%",
                }, ensure_ascii=False) + "\n")

        for _, row in scored.iterrows():
            obj = {"type": "item"}
            for col, val in row.to_dict().items():
                obj[col] = _jsonable(val)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return result


def print_open_qa_terminal_report(results_csv_path: str, report: dict = None) -> None:
    df = pd.read_csv(results_csv_path)
    scored = score_vqa_open(df)
    avg_wbss = float(scored["wbss"].mean() * 100)
    judge_str = ""
    if report and "judge_accuracy_pct" in report:
        judge_str = f"  LLM-Judge: {report['judge_accuracy_pct']:.2f}%"
    print(f"  WBSS: {avg_wbss:.2f}% ({len(scored)} questions){judge_str}")


# ===========================================================================
# CLI entry-point (extended)
# ===========================================================================

def _load_client_from_config():
    import yaml, os as _os
    cfg_path = "config.yaml" if _os.path.exists("config.yaml") else "config.default.yaml"
    with open(cfg_path) as _f:
        cfg = yaml.safe_load(_f)
    from core.client import MedicalLLMClient
    return MedicalLLMClient(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate benchmark results CSV.")
    parser.add_argument(
        "csv",
        nargs="?",
        default="results/benchmark_results.csv",
        help="Path to CSV (default: results/benchmark_results.csv)",
    )
    parser.add_argument(
        "--out",
        default="results/benchmark_report.jsonl",
        help="Output JSONL report path (default: results/benchmark_report.jsonl)",
    )
    parser.add_argument(
        "--type",
        dest="eval_type",
        choices=["mcq", "vqa", "extraction", "open_qa"],
        default="mcq",
        help="Evaluation type: mcq | vqa | extraction | open_qa",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        default=False,
        help="Run LLM-as-a-Judge for open-ended answers (requires config.yaml + live LLM).",
    )
    args = parser.parse_args()

    if args.eval_type == "mcq":
        report = write_report_jsonl(args.csv, out_path=args.out)
        print(f"Accuracy: {report['accuracy_pct']:.2f}%")
        print(f"Wrote: {report['path']}")

    elif args.eval_type == "vqa":
        client = _load_client_from_config() if args.judge else None
        report = write_vqa_report_jsonl(args.csv, out_path=args.out, client=client, run_judge=args.judge)
        print_vqa_terminal_report(args.csv)
        print(f"Wrote: {report['path']}")

    elif args.eval_type == "extraction":
        report = write_extraction_report_jsonl(args.csv, out_path=args.out)
        print_extraction_terminal_report(args.csv)
        print(f"Micro F1: {report['micro_f1_pct']:.2f}%")
        print(f"Wrote: {report['path']}")

    elif args.eval_type == "open_qa":
        client = _load_client_from_config() if args.judge else None
        report = write_open_qa_report_jsonl(args.csv, out_path=args.out, client=client, run_judge=args.judge)
        print_open_qa_terminal_report(args.csv)
        print(f"Wrote: {report['path']}")


if __name__ == "__main__":
    main()
