import argparse
import re
import string
from functools import lru_cache
from typing import Optional
import json

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
    logger=None,
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

    if logger:
        rows = int(metrics.loc[metrics["metric"] == "rows", "value"].iloc[0]) if not metrics.empty else len(df)
        parsed_model = int(metrics.loc[metrics["metric"] == "parsed_model_answer", "value"].iloc[0]) if not metrics.empty else 0
        logger.verbose(f"\n--- MCQ Evaluation ---")
        logger.verbose(f"Total: {rows}  Parsed: {parsed_model}  Accuracy: {accuracy_pct:.2f}%")

        # Answer distribution
        logger.verbose("\nAnswer distribution:")
        for _, row in dist.iterrows():
            logger.verbose(f"  {row['answer']:>8}  {row['count']:>5}  ({row['pct']:.1f}%)")

        # Confusion matrix
        logger.verbose("\nConfusion matrix (correct → model):")
        header = "       " + "".join(f"{col:>8}" for col in conf.columns)
        logger.verbose(header)
        for correct in conf.index:
            row_str = f"  {correct:>4}  " + "".join(f"{int(conf.loc[correct, col]):>8}" for col in conf.columns)
            logger.verbose(row_str)

        # Wrong examples (up to 20)
        wrong = scored[~scored["is_correct"]].head(20)
        if not wrong.empty:
            logger.verbose(f"\nWrong examples (first {len(wrong)}):")
            for _, row in wrong.iterrows():
                q_short = str(row.get("question", ""))[:80]
                logger.verbose(
                    f"  [{row.get('id')}] correct={row['correct_answer_norm']}  "
                    f"model={row['model_answer_norm']}  raw={str(row.get('model_answer',''))[:30]!r}\n"
                    f"    Q: {q_short}"
                )

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
    """
    Score MCQ rows in a VQA results CSV.

    Two modes depending on whether reference_answer is a letter or text:
    - Letter reference (e.g. RadImageNet-VQA): extract letter from both sides, compare.
    - Text reference (e.g. RadBench): model picks a letter, look up its text value via
      options_json, compare text to reference case-insensitively.
    """
    import json as _json
    df = df.copy()

    def _score_row(row):
        ref = str(row.get("reference_answer") or "").strip()
        model_raw = str(row.get("model_answer") or "").strip()
        model_letter = extract_choice(model_raw)

        # Case 1: reference is a single letter → classic letter comparison
        if len(ref) == 1 and ref.upper() in "ABCDE":
            return model_letter == ref.upper()

        # Case 2: reference is text → map model letter → text via options_json
        options_raw = row.get("options_json") or "[]"
        try:
            options = _json.loads(options_raw) if isinstance(options_raw, str) else (options_raw or [])
        except Exception:
            options = []

        if model_letter and options:
            letter_map = {o["key"].upper(): str(o["value"]).strip().lower()
                          for o in options if isinstance(o, dict) and "key" in o and "value" in o}
            model_text = letter_map.get(model_letter, "")
            return model_text == ref.strip().lower()

        # Fallback: direct text normalisation
        return _normalise_text(model_raw) == _normalise_text(ref)

    df["correct_answer_norm"] = df["reference_answer"].map(
        lambda r: r if (len(str(r).strip()) == 1 and str(r).strip().upper() in "ABCDE") else str(r).strip()
    )
    df["model_answer_norm"] = df["model_answer"].map(extract_choice)
    df["is_correct"] = df.apply(_score_row, axis=1)
    return df


def _wbss(prediction: str, reference: str) -> float:
    """
    Word-Based Semantic Similarity (WBSS) via Wu-Palmer similarity on WordNet.
    Used for VQA-Med-2019, RadImageNet-VQA, RadBench open, and RadioRAG.
    Requires: nltk + nltk.download('wordnet') + nltk.download('omw-1.4')
    Returns 0.0 if WordNet is unavailable.
    """
    try:
        from nltk.corpus import wordnet as wn

        @lru_cache(maxsize=2048)
        def _synsets(word):
            return wn.synsets(word)

        pred_tokens = _normalise_text(prediction).split()
        ref_tokens = _normalise_text(reference).split()
        if not pred_tokens or not ref_tokens:
            return 0.0

        def best_wup(word, candidates):
            syns_w = _synsets(word)
            if not syns_w:
                return 0.0
            best = 0.0
            for cand in candidates:
                for sc in _synsets(cand):
                    for sw in syns_w:
                        sim = sw.wup_similarity(sc)
                        if sim and sim > best:
                            best = sim
            return best

        p2r = sum(best_wup(w, tuple(ref_tokens)) for w in pred_tokens) / len(pred_tokens)
        r2p = sum(best_wup(w, tuple(pred_tokens)) for w in ref_tokens) / len(ref_tokens)
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
    import multiprocessing as _mp
    df = df.copy()
    pairs = list(zip(df["model_answer"].astype(str), df["reference_answer"].astype(str)))
    workers = min(_mp.cpu_count(), 8)
    with _mp.Pool(workers) as pool:
        df["wbss"] = pool.starmap(_wbss, pairs)
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
    logger=None,
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
            scored_mcq = score_vqa_mcq(mcq_df)
            accuracy = float(scored_mcq["is_correct"].mean() * 100)
            results["mcq_accuracy_pct"] = round(accuracy, 2)
            results["mcq_rows"] = len(scored_mcq)
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
            results["yes_no_rows"] = len(scored_yn)
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
            results["open_rows"] = len(scored_open)

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

    if logger:
        logger.verbose("\n--- VQA Evaluation ---")
        if not mcq_df.empty:
            logger.verbose(f"MCQ: {results.get('mcq_rows', 0)} questions  Accuracy: {results.get('mcq_accuracy_pct', 0):.2f}%")
            wrong_mcq = scored_mcq[~scored_mcq["is_correct"]].head(10)
            if not wrong_mcq.empty:
                logger.verbose(f"  Wrong MCQ examples (first {len(wrong_mcq)}):")
                for _, row in wrong_mcq.iterrows():
                    logger.verbose(
                        f"    [{row.get('id')}] correct={row['correct_answer_norm']}  "
                        f"model={row['model_answer_norm']}  raw={str(row.get('model_answer',''))[:30]!r}"
                    )
        if not yes_no_df.empty:
            logger.verbose(f"Yes/No: {results.get('yes_no_rows', 0)} questions  Accuracy: {results.get('yes_no_accuracy_pct', 0):.2f}%")
        if not open_df.empty:
            logger.verbose(
                f"Open: {results.get('open_rows', 0)} questions  WBSS: {results.get('open_wbss_pct', 0):.2f}%"
                + (f"  LLM-Judge: {results.get('open_judge_accuracy_pct', 0):.2f}%" if "open_judge_accuracy_pct" in results else "")
            )
            # Bottom-20 open questions by WBSS
            bottom = scored_open.nsmallest(20, "wbss")
            logger.verbose(f"  Bottom {len(bottom)} open answers by WBSS:")
            for _, row in bottom.iterrows():
                q_short = str(row.get("question", ""))[:60]
                ref_short = str(row.get("reference_answer", ""))[:40]
                ans_short = str(row.get("model_answer", ""))[:40]
                judge = f"  judge={int(row['judge_correct'])}" if "judge_correct" in row and pd.notna(row.get("judge_correct")) else ""
                logger.verbose(
                    f"    [{row.get('id')}] wbss={row['wbss']:.3f}{judge}\n"
                    f"      Q:   {q_short}\n"
                    f"      Ref: {ref_short}\n"
                    f"      Ans: {ans_short}"
                )

    return results


def print_vqa_terminal_report(results_csv_path: str, report: dict = None) -> None:
    r = report or {}
    parts = []

    if "mcq_accuracy_pct" in r:
        parts.append(f"MCQ Accuracy: {r['mcq_accuracy_pct']:.2f}% ({r.get('mcq_rows', '?')} questions)")
    if "yes_no_accuracy_pct" in r:
        parts.append(f"Yes/No Accuracy: {r['yes_no_accuracy_pct']:.2f}% ({r.get('yes_no_rows', '?')} questions)")
    if "open_wbss_pct" in r:
        judge_str = f"  LLM-Judge: {r['open_judge_accuracy_pct']:.2f}%" if "open_judge_accuracy_pct" in r else ""
        parts.append(f"Open WBSS: {r['open_wbss_pct']:.2f}% ({r.get('open_rows', '?')} questions){judge_str}")

    if not parts:
        # fallback: recompute from CSV (e.g. when called standalone via CLI)
        df = pd.read_csv(results_csv_path)
        q_type_col = "question_type" if "question_type" in df.columns else None
        mcq_df = df[df[q_type_col] == "mcq"] if q_type_col else pd.DataFrame()
        yes_no_df = df[df[q_type_col] == "yes_no"] if q_type_col else pd.DataFrame()
        open_df = df[df[q_type_col] == "open"] if q_type_col else df
        if not mcq_df.empty:
            acc = float(score_vqa_mcq(mcq_df)["is_correct"].mean() * 100)
            parts.append(f"MCQ Accuracy: {acc:.2f}% ({len(mcq_df)} questions)")
        if not yes_no_df.empty:
            yn_acc = float(yes_no_df.apply(
                lambda row: _normalise_text(str(row["model_answer"])) == _normalise_text(str(row["reference_answer"])), axis=1
            ).mean() * 100)
            parts.append(f"Yes/No Accuracy: {yn_acc:.2f}% ({len(yes_no_df)} questions)")
        if not open_df.empty:
            scored = score_vqa_open(open_df)
            parts.append(f"Open WBSS: {float(scored['wbss'].mean() * 100):.2f}% ({len(scored)} questions)")

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
    logger=None,
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

    if logger:
        logger.verbose("\n--- Extraction Evaluation ---")
        logger.verbose(f"Micro F1: {micro_f1:.2f}%  P: {micro_p:.2f}%  R: {micro_r:.2f}%  ({len(scored)} texts)")
        # Worst 20 by per-item F1
        scored["item_f1"] = scored.apply(
            lambda r: (2 * r["tp"] / (2 * r["tp"] + r["fp"] + r["fn"])) if (2 * r["tp"] + r["fp"] + r["fn"]) > 0 else 0.0,
            axis=1,
        )
        worst = scored.nsmallest(20, "item_f1")
        logger.verbose(f"  Worst {len(worst)} items by item F1:")
        for _, row in worst.iterrows():
            logger.verbose(
                f"    [{row.get('id')}] f1={row['item_f1']:.3f}  tp={row['tp']}  fp={row['fp']}  fn={row['fn']}\n"
                f"      Ref: {str(row.get('reference_entities',''))[:80]}\n"
                f"      Got: {str(row.get('model_entities',''))[:80]}"
            )

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
    logger=None,
) -> dict:
    """
    Evaluate open-ended QA results (RadioRAG).

    Automatic metric: WBSS.
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

    if logger:
        logger.verbose("\n--- Open QA Evaluation (RadioRAG) ---")
        logger.verbose(
            f"WBSS: {result['wbss_pct']:.2f}%  ({len(scored)} questions)"
            + (f"  LLM-Judge: {result.get('judge_accuracy_pct', 0):.2f}%" if "judge_accuracy_pct" in result else "")
        )
        # Judge-incorrect examples (up to 20)
        if "judge_correct" in scored.columns:
            incorrect = scored[scored["judge_correct"] == 0].head(20)
            if not incorrect.empty:
                logger.verbose(f"  Judge-incorrect examples (first {len(incorrect)}):")
                for _, row in incorrect.iterrows():
                    q_short = str(row.get("question", ""))[:60]
                    ref_short = str(row.get("reference_answer", ""))[:50]
                    ans_short = str(row.get("model_answer", ""))[:50]
                    logger.verbose(
                        f"    [{row.get('id')}] wbss={row['wbss']:.3f}\n"
                        f"      Q:   {q_short}\n"
                        f"      Ref: {ref_short}\n"
                        f"      Ans: {ans_short}"
                    )
        else:
            # Bottom-20 by WBSS when no judge
            bottom = scored.nsmallest(20, "wbss")
            logger.verbose(f"  Bottom {len(bottom)} answers by WBSS:")
            for _, row in bottom.iterrows():
                q_short = str(row.get("question", ""))[:60]
                ref_short = str(row.get("reference_answer", ""))[:50]
                ans_short = str(row.get("model_answer", ""))[:50]
                logger.verbose(
                    f"    [{row.get('id')}] wbss={row['wbss']:.3f}\n"
                    f"      Q:   {q_short}\n"
                    f"      Ref: {ref_short}\n"
                    f"      Ans: {ans_short}"
                )

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
