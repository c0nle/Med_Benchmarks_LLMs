import argparse
import re
import string
from functools import lru_cache
from typing import Optional
import json
import os

import pandas as pd


CHOICE_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)


def extract_choice(value, valid_keys: str = "ABCDE") -> Optional[str]:
    """
    Parse the chosen option letter from a model reply.

    Matches only uppercase letters in the original text (so the words "I" and "a"
    are not mistaken for options), in order of reliability:
    bare letter → "answer is/: X" → leading "X)", "(X)", "**X**", "X." / "X:" → last
    standalone capital option letter.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text.startswith("Error:"):
        return None
    keys = re.escape(valid_keys.upper())

    bare = text.strip("*()[]. :").strip()
    if len(bare) == 1 and bare.upper() in valid_keys.upper():
        return bare.upper()

    patterns = [
        rf"(?i:answer|choice)\s*(?:is|:)?\s*\**\(?([{keys}])\)?\**(?![A-Za-z])",
        rf"^\s*\**\(?([{keys}])\)?\**\s*[\).:\-]",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)

    # Last standalone capital letter; "I" only counts when it is a valid key and
    # not followed by an apostrophe (I'd, I'm).
    candidates = [
        m.group(1) for m in re.finditer(rf"(?<![A-Za-z'])([{keys}])(?![A-Za-z'’])", text)
    ]
    candidates = [c for c in candidates if c != "I" or "I" in valid_keys.upper() and not re.search(r"\bI\s+(think|believe|would|am|choose)\b", text)]
    return candidates[-1] if candidates else None


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

    def _options(row):
        options_raw = row.get("options_json") or "[]"
        try:
            options = _json.loads(options_raw) if isinstance(options_raw, str) else (options_raw or [])
        except Exception:
            options = []
        return [o for o in options if isinstance(o, dict) and "key" in o and "value" in o]

    def _valid_keys(row):
        keys = "".join(str(o["key"]).upper() for o in _options(row))
        return keys or "ABCDE"

    def _score_row(row):
        ref = str(row.get("reference_answer") or "").strip()
        model_raw = str(row.get("model_answer") or "").strip()
        model_letter = extract_choice(model_raw, _valid_keys(row))

        # Case 1: reference is a single letter → classic letter comparison
        if len(ref) == 1 and ref.upper() in "ABCDE":
            return model_letter == ref.upper()

        # Case 2: reference is text → map model letter → text via options_json
        options = _options(row)

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
    df["model_answer_norm"] = df.apply(lambda r: extract_choice(r["model_answer"], _valid_keys(r)), axis=1)
    df["is_correct"] = df.apply(_score_row, axis=1)
    return df


def _wbss(prediction: str, reference: str) -> float:
    """
    Word-Based Semantic Similarity (WBSS) via Wu-Palmer similarity on WordNet.
    Used for VQA-Med-2019, RadImageNet-VQA, RadBench open, and RadioRAG.
    Requires: nltk + nltk.download('wordnet') + nltk.download('omw-1.4')
    Identical tokens score 1.0 even if they are not in WordNet (e.g. "t2", "cta").
    """
    from nltk.corpus import wordnet as wn

    @lru_cache(maxsize=2048)
    def _synsets(word):
        return wn.synsets(word)

    pred_tokens = _normalise_text(prediction).split()
    ref_tokens = _normalise_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    def best_wup(word, candidates):
        if word in candidates:
            return 1.0
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


def parse_judge_reply(raw) -> Optional[int]:
    """Judge verdict 0/1; None if the reply is an error or not a clear verdict."""
    text = str(raw or "")
    if text.startswith("Error:"):
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.fullmatch(r"[\s*'\"`]*([01])[\s*'\"`.]*", text)
    return int(m.group(1)) if m else None


def evaluate_vqa_with_judge(
    df: pd.DataFrame,
    client,
    cache_path: Optional[str] = None,
    workers: int = 8,
) -> pd.DataFrame:
    """
    LLM-as-a-Judge evaluation for open-ended VQA rows.

    Implements the binary correct/incorrect rubric used by RadImageNet-VQA
    (Butsanets et al., 2025, following Zheng et al., 2023):
    The judge receives the question, the ground-truth answer, and the model
    prediction and returns 1 (correct) or 0 (incorrect).

    Raw judge replies are appended to *cache_path* (CSV: id, judge_raw) as they
    arrive and reused on re-evaluation, so an interrupted run loses nothing.
    Model answers that are API errors are not sent to the judge (judge_correct=0).

    Returns a copy of df with 'judge_raw' and 'judge_correct' (0|1, NaN if unparsed).
    """
    import csv as _csv
    import threading
    from concurrent.futures import ThreadPoolExecutor

    df = df.copy()
    ids = df["id"].astype(str).tolist()

    cache: dict = {}
    if cache_path and os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        cached = pd.read_csv(cache_path, dtype=str).fillna("")
        cache = {r["id"]: r["judge_raw"] for _, r in cached.iterrows()
                 if parse_judge_reply(r["judge_raw"]) is not None}

    def _prompt(row):
        return (
            "You are a medical expert judge evaluating a model's answer to a radiology question.\n\n"
            f"Question: {row['question']}\n"
            f"Ground-truth answer: {row['reference_answer']}\n"
            f"Model answer: {row['model_answer']}\n\n"
            "Is the model answer medically correct and equivalent in meaning to the ground-truth answer?\n"
            "Reply with exactly '1' (correct) or '0' (incorrect). No other text."
        )

    todo = [(i, row) for i, (_, row) in zip(ids, df.iterrows())
            if i not in cache and not str(row["model_answer"]).startswith("Error:")]
    if todo:
        print(f"  LLM-Judge: {len(todo)} answers to judge ({len(cache)} cached)...")

    lock = threading.Lock()
    done = [0]
    cache_file = None
    writer = None
    if cache_path:
        new_file = not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 0)
        cache_file = open(cache_path, "a", newline="", encoding="utf-8")
        writer = _csv.DictWriter(cache_file, fieldnames=["id", "judge_raw"])
        if new_file:
            writer.writeheader()

    def _judge(item):
        item_id, row = item
        raw = client.ask_question(_prompt(row))
        with lock:
            cache[item_id] = raw
            if writer:
                writer.writerow({"id": item_id, "judge_raw": raw})
                cache_file.flush()
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"    judged {done[0]}/{len(todo)}")

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_judge, todo))
    finally:
        if cache_file:
            cache_file.close()

    df["judge_raw"] = [cache.get(i, "") for i in ids]
    df["judge_correct"] = [
        0 if str(ans).startswith("Error:") else parse_judge_reply(cache.get(i))
        for i, ans in zip(ids, df["model_answer"])
    ]
    return df


def _judge_summary(scored: pd.DataFrame) -> dict:
    """Judge accuracy over parsed verdicts plus counts, so unparsed ones are visible."""
    valid = scored["judge_correct"].dropna()
    out = {"n_judged": int(len(valid)), "n_judge_unparsed": int(len(scored) - len(valid))}
    if not valid.empty:
        out["judge_accuracy_pct"] = round(float(valid.mean() * 100), 2)
    if out["n_judge_unparsed"]:
        print(f"  WARNING: {out['n_judge_unparsed']} judge verdicts could not be parsed "
              f"(excluded from LLM-Judge accuracy)")
    return out


def _yes_no_token(text) -> Optional[str]:
    words = _normalise_text(str(text or "")).split()
    return words[0] if words and words[0] in ("yes", "no") else None


def _judge_cache_path(results_csv_path: str) -> str:
    base = results_csv_path[:-len("_results.csv")] if results_csv_path.endswith("_results.csv") else results_csv_path
    return base + "_judge_cache.csv"


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

    # Judge before opening the report file, so an interrupted judge run never
    # leaves an empty report behind (verdicts are cached next to the results CSV).
    scored_open = None
    if not open_df.empty:
        scored_open = score_vqa_open(open_df)
        scored_open["exact_match"] = [
            _exact_match(str(p), str(r))
            for p, r in zip(scored_open["model_answer"], scored_open["reference_answer"])
        ]
        if run_judge and client is not None:
            scored_open = evaluate_vqa_with_judge(
                scored_open, client, cache_path=_judge_cache_path(results_csv_path)
            )

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
        # RadImageNet-VQA "checks for the expected token": compare the first word.
        if not yes_no_df.empty:
            scored_yn = yes_no_df.copy()
            scored_yn["is_correct"] = scored_yn.apply(
                lambda r: _yes_no_token(r["model_answer"]) == _yes_no_token(r["reference_answer"])
                and _yes_no_token(r["reference_answer"]) is not None,
                axis=1,
            )
            yn_acc = float(scored_yn["is_correct"].mean() * 100)
            results["yes_no_accuracy_pct"] = round(yn_acc, 2)
            results["yes_no_rows"] = len(scored_yn)
            f.write(json.dumps({"type": "metric", "subset": "yes_no", "metric": "accuracy_pct", "value": round(yn_acc, 2), "note": "first token yes/no, RadImageNet-VQA closed-ended task"}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"type": "metric", "subset": "yes_no", "metric": "rows", "value": len(scored_yn)}, ensure_ascii=False) + "\n")
            for _, row in scored_yn.iterrows():
                obj = {"type": "item", "subset": "yes_no"}
                for col, val in row.to_dict().items():
                    obj[col] = _jsonable(val)
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # --- Open-ended sub-results ---
        if scored_open is not None:
            avg_wbss = float(scored_open["wbss"].mean() * 100)
            results["open_wbss_pct"] = round(avg_wbss, 2)
            results["open_rows"] = len(scored_open)
            em = round(float(scored_open["exact_match"].mean() * 100), 2)
            results["open_exact_match_pct"] = em
            f.write(json.dumps({"type": "metric", "subset": "open", "metric": "exact_match_pct", "value": em, "note": "normalised exact match (official VQA-Med-2019 accuracy)"}, ensure_ascii=False) + "\n")

            f.write(json.dumps({"type": "metric", "subset": "open", "metric": "wbss_pct", "value": round(avg_wbss, 2), "note": "Wu-Palmer semantic similarity via WordNet"}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"type": "metric", "subset": "open", "metric": "rows", "value": len(scored_open)}, ensure_ascii=False) + "\n")

            # LLM-as-a-Judge (binary 0/1) — RadImageNet-VQA open-ended metric
            if "judge_correct" in scored_open.columns:
                js = _judge_summary(scored_open)
                results["open_n_judge_unparsed"] = js["n_judge_unparsed"]
                if "judge_accuracy_pct" in js:
                    results["open_judge_accuracy_pct"] = js["judge_accuracy_pct"]
                f.write(json.dumps({"type": "metric", "subset": "open", "metric": "llm_judge_accuracy_pct", "value": js.get("judge_accuracy_pct"), "n_judged": js["n_judged"], "n_judge_unparsed": js["n_judge_unparsed"], "note": "binary correct/incorrect over parsed verdicts, RadImageNet-VQA primary metric"}, ensure_ascii=False) + "\n")

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
        if scored_open is not None:
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
        unparsed = r.get("open_n_judge_unparsed")
        unparsed_str = f" ({unparsed} unparsed)" if unparsed else ""
        parts.append(
            f"Open ({r.get('open_rows', '?')} questions): Exact {r.get('open_exact_match_pct', 0):.2f}%  "
            f"WBSS {r['open_wbss_pct']:.2f}%{judge_str}{unparsed_str}"
        )

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
        scored = evaluate_vqa_with_judge(scored, client, cache_path=_judge_cache_path(results_csv_path))

    avg_wbss = float(scored["wbss"].mean() * 100)

    result = {
        "path": out_path,
        "wbss_pct": round(avg_wbss, 2),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "metric", "metric": "rows", "value": len(scored)}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"type": "metric", "metric": "wbss_pct", "value": round(avg_wbss, 2)}, ensure_ascii=False) + "\n")

        if "judge_correct" in scored.columns:
            js = _judge_summary(scored)
            result["n_judge_unparsed"] = js["n_judge_unparsed"]
            if "judge_accuracy_pct" in js:
                result["judge_accuracy_pct"] = js["judge_accuracy_pct"]
            f.write(json.dumps({
                "type": "metric",
                "metric": "llm_judge_accuracy_pct",
                "value": js.get("judge_accuracy_pct"),
                "n_judged": js["n_judged"],
                "n_judge_unparsed": js["n_judge_unparsed"],
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
        choices=["mcq", "vqa", "extraction", "open_qa", "mamma_extraction", "arm_extraction"],
        default="mcq",
        help="Evaluation type",
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
        print_vqa_terminal_report(args.csv, report=report)
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

    elif args.eval_type == "mamma_extraction":
        report = write_mamma_extraction_report_jsonl(args.csv, out_path=args.out)
        print_mamma_extraction_terminal_report(args.csv, report=report)
        print(f"Wrote: {report['path']}")

    elif args.eval_type == "arm_extraction":
        report = write_arm_extraction_report_jsonl(args.csv, out_path=args.out)
        print_arm_extraction_terminal_report(args.csv, report=report)
        print(f"Wrote: {report['path']}")



# ===========================================================================
# Mamma-MRT Label Extraction Evaluation
# ===========================================================================

import json as _json
import random as _random
import re as _re
from collections import Counter as _Counter
from functools import lru_cache as _lru_cache
from pathlib import Path as _Path

_MAMMA_NORM_PATH = _Path("config/mamma_normalization.yaml")


@_lru_cache(maxsize=1)
def _load_mamma_norm() -> dict:
    """Lädt Normalisierungs-YAML (gecacht). Gibt leeres Dict bei fehlendem File zurück."""
    if not _MAMMA_NORM_PATH.exists():
        return {}
    import yaml as _yaml
    with open(_MAMMA_NORM_PATH, encoding="utf-8") as f:
        return _yaml.safe_load(f) or {}


def _build_normalizer(mapping: dict) -> dict:
    """Baut invertierten Dict: variante.lower().strip() → kanonische Form."""
    inv: dict = {}
    for canonical, variants in (mapping or {}).items():
        key = str(canonical).lower().strip()
        inv[key] = str(canonical)
        for v in (variants or []):
            inv[str(v).lower().strip()] = str(canonical)
    return inv


def _normalize_val(value, normalizer: dict):
    """Normalisiert einen Wert via Normalizer-Dict."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s in ("nan", "None", "?"):
        return None
    return normalizer.get(s.lower(), s.lower())


def _normalize_birads(value, normalizer: dict, birads6_handling: str = "map_to_5"):
    """Normalisiert BIRADS; extrahiert führende Ziffer aus Texten wie '6 nachgewiesene...'."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s in ("nan", "None", "?", "ERROR"):
        return None

    # Erst über YAML-Mapping normalisieren
    normed = normalizer.get(s.lower())

    # Fallback: führende Ziffer extrahieren (z.B. "4 Suspekt..." → "4")
    if normed is None:
        m = _re.search(r"(?<!\d)([1-6])(?!\d)", s)
        if m:
            normed = m.group(1)
        else:
            # Numerisch direkt?
            try:
                normed = str(int(float(s)))
            except (ValueError, TypeError):
                normed = None

    if normed == "6" and birads6_handling == "map_to_5":
        normed = "5"
    return normed


def _normalize_acr(value, normalizer: dict, acr_range: str = "min"):
    """
    Normalisiert ACR/BPE-Wert.
    Behandelt Bereiche wie '1 bis 2' per acr_range-Option.
    Gibt (kanonischer Wert, is_range_error) zurück.
    """
    if value is None:
        return None, False
    s = str(value).strip()
    if not s or s in ("nan", "None", "?", "ERROR"):
        return None, False

    # Bereich? z.B. "1 bis 2", "2-3", "1 to 2"
    m_range = _re.search(r"(?<!\d)([1-4])\s*(?:bis|to|-|–)\s*([1-4])(?!\d)", s, _re.IGNORECASE)
    if m_range:
        a, b = int(m_range.group(1)), int(m_range.group(2))
        if acr_range == "min":
            s = str(min(a, b))
        elif acr_range == "max":
            s = str(max(a, b))
        else:  # "error"
            return None, True

    normed = normalizer.get(s.lower())
    if normed is None:
        # Führende Ziffer extrahieren
        m = _re.search(r"(?<!\d)([1-4])(?!\d)", s)
        if m:
            normed = m.group(1)
        else:
            try:
                v = int(float(s))
                normed = str(v) if 1 <= v <= 4 else None
            except (ValueError, TypeError):
                normed = None
    return normed, False


def _multiset_prf(gt_list: list, pred_list: list) -> tuple[int, int, int]:
    """Multiset TP/FP/FN für Läsionslisten."""
    gt_c   = _Counter(gt_list)
    pred_c = _Counter(pred_list)
    tp = sum(min(gt_c[k], pred_c.get(k, 0)) for k in gt_c)
    fp = sum(max(0, pred_c[k] - gt_c.get(k, 0)) for k in pred_c)
    fn = sum(max(0, gt_c[k] - pred_c.get(k, 0)) for k in gt_c)
    return tp, fp, fn


def _bootstrap_ci(values: list, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float, float]:
    """95%-Bootstrap-CI auf Report-Ebene. Gibt (point, lo, hi) zurück."""
    if not values:
        return 0.0, 0.0, 0.0
    n = len(values)
    if n == 1:
        v = float(values[0])
        return v, v, v
    rng = _random.Random(42)
    boot = sorted(
        sum(values[rng.randint(0, n - 1)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    lo = boot[int(n_boot * alpha / 2)]
    hi = boot[int(n_boot * (1 - alpha / 2)) - 1]
    return sum(values) / n, lo, hi


def _bootstrap_micro_f1_ci(counts: list, n_boot: int = 1000, alpha: float = 0.05):
    """
    95% CI for micro-F1: resample reports, sum their (tp, fp, fn), recompute micro-F1.
    Returns (lo, hi) as fractions.
    """
    if not counts:
        return 0.0, 0.0

    def _f1(sample):
        tp = sum(c[0] for c in sample)
        fp = sum(c[1] for c in sample)
        fn = sum(c[2] for c in sample)
        denom = 2 * tp + fp + fn
        return 2 * tp / denom if denom > 0 else 0.0

    n = len(counts)
    rng = _random.Random(42)
    boot = sorted(_f1([counts[rng.randint(0, n - 1)] for _ in range(n)]) for _ in range(n_boot))
    return boot[int(n_boot * alpha / 2)], boot[int(n_boot * (1 - alpha / 2)) - 1]


def _categorical_metrics(y_true: list, y_pred: list) -> dict:
    """Accuracy, Macro-F1 und Konfusionsmatrix aus zwei parallelen Listen."""
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "confusion": {}}

    # Macro over real GT classes only; sentinels like "__missing__" are not classes.
    classes = sorted(c for c in set(y_true) if not str(c).startswith("__"))
    f1s = []
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)

    accuracy  = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    macro_f1  = sum(f1s) / len(f1s) if f1s else 0.0

    confusion: dict = {}
    for t, p in zip(y_true, y_pred):
        confusion.setdefault(t, {}).setdefault(p, 0)
        confusion[t][p] += 1

    return {"accuracy": accuracy, "macro_f1": macro_f1, "confusion": confusion}


def _normalize_lesion_list(raw_json: str, lesion_norm: dict) -> list:
    """JSON-String → normalisierte Läsionsliste."""
    try:
        items = _json.loads(raw_json) if isinstance(raw_json, str) else (raw_json or [])
    except Exception:
        return []
    if not isinstance(items, list):
        return []
    result = []
    for item in items:
        s = str(item).strip()
        normed = lesion_norm.get(s.lower(), s.lower())
        if normed:
            result.append(normed)
    return result


def write_mamma_extraction_report_jsonl(
    results_csv_path: str,
    out_path: str,
    config: dict = None,
    logger=None,
) -> dict:
    """
    Wertet Mamma-MRT-Extraktions-CSV aus.

    Config-Optionen (unter task_settings.label_extraction_mamma):
        acr_interpretation : "bpe" (Standard) | "density"
        acr_range          : "min" (Standard) | "max" | "error"
        birads6_handling   : "map_to_5" (Standard) | "keep"
        gt_empty_ext_present: "ignore" (Standard) | "fp"  – gilt für Felder und Läsions-Seiten

    Metriken pro kategoriales Feld: Accuracy, Macro-F1, Konfusionsmatrix, Bootstrap-CI.
    Läsionen pro Seite: Hauptmetrik = Typ-Menge (welche Läsionstypen kommen vor),
    zusätzlich Anzahl-Sicht (Multiset, eine Zeile pro Läsion); je Micro-P/R/F1, Exact Match, Bootstrap-CI.
    Zähler: n_gesamt, n_bewertet, n_ignoriert_nur_EXT, n_beide_leer, n_parse_error.
    """
    cfg_raw = {}
    if config:
        cfg_raw = config.get("task_settings", {}).get("label_extraction_mamma", {})

    acr_range          = cfg_raw.get("acr_range",            "min")
    birads6_handling   = cfg_raw.get("birads6_handling",      "map_to_5")
    gt_empty_fp        = cfg_raw.get("gt_empty_ext_present",  "ignore") == "fp"

    norm         = _load_mamma_norm()
    meno_norm    = _build_normalizer(norm.get("menopause",    {}))
    birads_norm  = _build_normalizer(norm.get("birads",       {}))
    acr_norm     = _build_normalizer(norm.get("acr",          {}))
    lesion_norm  = _build_normalizer(norm.get("lesion_types", {}))

    df = pd.read_csv(results_csv_path, dtype=str).fillna("")

    n_gesamt     = len(df)
    n_parse_error = int((df["parse_error"].str.lower() == "true").sum())

    # Initialisiere Sammelstrukturen
    fields = ["menopause", "birads_li", "birads_re", "acr_li", "acr_re"]
    field_data: dict[str, dict] = {
        f: {"y_true": [], "y_pred": [], "n_bewertet": 0,
            "n_ignoriert_nur_EXT": 0, "n_beide_leer": 0,
            "per_item_correct": []}
        for f in fields
    }

    # Two lesion views per side:
    #   "type"  (main metric): which lesion types occur on the side (set comparison)
    #   "count" (secondary):   one entry per lesion (multiset); GT has one row per
    #                          lesion, so multifocal findings count several times
    lesion_sides = {
        (side, mode): {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "n": 0, "ignored": 0,
                       "per_item_counts": []}
        for side in ("lesions_li", "lesions_re") for mode in ("type", "count")
    }

    # Reihe für Reihe verarbeiten
    for _, row in df.iterrows():
        is_parse_err = row["parse_error"].lower() == "true"

        # ─── Kategoriale Felder ───
        for field in fields:
            prefix = field.split("_")[0]  # menopause / birads / acr
            gt_raw  = row.get(f"gt_{field}",    "")
            ext_raw = row.get(f"model_{field}", "")
            fd      = field_data[field]

            if prefix == "menopause":
                gt_n  = _normalize_val(gt_raw,  meno_norm)
                ext_n = _normalize_val(ext_raw, meno_norm) if not is_parse_err else None
            elif prefix == "birads":
                gt_n  = _normalize_birads(gt_raw,  birads_norm, birads6_handling)
                ext_n = _normalize_birads(ext_raw, birads_norm, birads6_handling) if not is_parse_err else None
            else:  # acr
                gt_n,  _   = _normalize_acr(gt_raw,  acr_norm, acr_range)
                ext_n, rng_err = _normalize_acr(ext_raw, acr_norm, acr_range) if not is_parse_err else (None, False)
                if rng_err:
                    ext_n = None

            gt_empty  = gt_n is None
            ext_empty = ext_n is None

            if gt_empty and ext_empty:
                fd["n_beide_leer"] += 1
            elif gt_empty and not ext_empty:
                fd["n_ignoriert_nur_EXT"] += 1
                if gt_empty_fp:
                    fd["y_true"].append("__empty__")
                    fd["y_pred"].append(ext_n)
                    fd["n_bewertet"] += 1
                    fd["per_item_correct"].append(0)
            elif not gt_empty and ext_empty:
                fd["y_true"].append(gt_n)
                fd["y_pred"].append("__missing__")
                fd["n_bewertet"] += 1
                fd["per_item_correct"].append(0)
            else:
                correct = int(gt_n == ext_n)
                fd["y_true"].append(gt_n)
                fd["y_pred"].append(ext_n)
                fd["n_bewertet"] += 1
                fd["per_item_correct"].append(correct)

        # ─── Läsionen (Typ-Menge + Multiset) ───
        for side_key in ("lesions_li", "lesions_re"):
            gt_side  = row.get(f"gt_{side_key}",    "[]")
            ext_side = row.get(f"model_{side_key}", "[]")

            gt_list  = _normalize_lesion_list(gt_side,  lesion_norm)
            ext_list = _normalize_lesion_list(ext_side, lesion_norm) if not is_parse_err else []

            for mode, (g, p) in (
                ("type",  (sorted(set(gt_list)), sorted(set(ext_list)))),
                ("count", (gt_list, ext_list)),
            ):
                sd = lesion_sides[(side_key, mode)]
                # The GT lesion table only lists lesions with histology/follow-up, so a
                # side without GT lesions is not scored (same rule as for empty GT fields)
                if not g and not gt_empty_fp:
                    sd["ignored"] += 1
                    continue
                tp, fp, fn = _multiset_prf(g, p)
                sd["tp"] += tp
                sd["fp"] += fp
                sd["fn"] += fn
                sd["n"]  += 1
                sd["per_item_counts"].append((tp, fp, fn))
                sd["exact"] += int(_Counter(g) == _Counter(p))

    # ─── Metriken schreiben ───────────────────────────────────────────────────
    result: dict = {"path": out_path}

    with open(out_path, "w", encoding="utf-8") as f:
        def _w(obj):
            f.write(_json.dumps(obj, ensure_ascii=False) + "\n")

        _w({"type": "metric", "metric": "n_gesamt",     "value": n_gesamt})
        _w({"type": "metric", "metric": "n_parse_error","value": n_parse_error})

        for field in fields:
            fd = field_data[field]
            yt, yp = fd["y_true"], fd["y_pred"]
            cm_res = _categorical_metrics(yt, yp)
            # Nothing to score → undefined (None), not 0%
            acc  = round(cm_res["accuracy"]  * 100, 2) if yt else None
            mf1  = round(cm_res["macro_f1"]  * 100, 2) if yt else None

            _, acc_lo, acc_hi = _bootstrap_ci(fd["per_item_correct"])
            acc_lo = round(acc_lo * 100, 2) if yt else None
            acc_hi = round(acc_hi * 100, 2) if yt else None

            if yt:
                result[f"{field}_accuracy_pct"] = acc
                result[f"{field}_macro_f1_pct"] = mf1

            _w({"type": "metric", "field": field, "metric": "n_bewertet",
                "value": fd["n_bewertet"]})
            _w({"type": "metric", "field": field, "metric": "n_ignoriert_nur_EXT",
                "value": fd["n_ignoriert_nur_EXT"]})
            _w({"type": "metric", "field": field, "metric": "n_beide_leer",
                "value": fd["n_beide_leer"]})
            _w({"type": "metric", "field": field, "metric": "accuracy_pct",
                "value": acc, "ci_lo": acc_lo, "ci_hi": acc_hi})
            _w({"type": "metric", "field": field, "metric": "macro_f1_pct", "value": mf1})

            for gt_cls, preds in cm_res["confusion"].items():
                for pred_cls, count in preds.items():
                    if count:
                        _w({"type": "confusion", "field": field,
                            "gt": gt_cls, "model": pred_cls, "count": count})

        for (side_key, mode), sd in lesion_sides.items():
            tp, fp, fn = sd["tp"], sd["fp"], sd["fn"]
            micro_p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            micro_r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
            exact_pct = (sd["exact"] / sd["n"] * 100) if sd["n"] > 0 else 0.0
            f1_defined = (tp + fp + fn) > 0

            f1_lo, f1_hi = _bootstrap_micro_f1_ci(sd["per_item_counts"])

            # Main metric (type) keeps the plain key names; count view is prefixed.
            prefix = side_key if mode == "type" else f"{side_key}_count"
            if f1_defined:
                result[f"{prefix}_micro_f1_pct"] = round(micro_f1 * 100, 2)
            result[f"{prefix}_exact_match_pct"] = round(exact_pct, 2)

            def _pct(x):
                return round(x * 100, 2) if f1_defined else None

            base = {"type": "metric", "field": side_key, "lesion_view": mode,
                    "n_sides": sd["n"], "n_ignored_gt_empty": sd["ignored"]}
            _w({**base, "metric": "micro_precision_pct", "value": _pct(micro_p)})
            _w({**base, "metric": "micro_recall_pct", "value": _pct(micro_r)})
            _w({**base, "metric": "micro_f1_pct", "value": _pct(micro_f1),
                "ci_lo": _pct(f1_lo), "ci_hi": _pct(f1_hi),
                "note": "CI: bootstrap over reports, micro-F1 recomputed per resample"})
            _w({**base, "metric": "exact_match_pct", "value": round(exact_pct, 2)})

        for _, row in df.iterrows():
            obj = {"type": "item"}
            for col, val in row.to_dict().items():
                obj[col] = _jsonable(val)
            f.write(_json.dumps(obj, ensure_ascii=False) + "\n")

    if logger:
        logger.verbose("\n--- Mamma-MRT Extraction Evaluation ---")
        logger.verbose(f"n_gesamt={n_gesamt}  n_parse_error={n_parse_error}")
        for field in fields:
            acc = result.get(f"{field}_accuracy_pct", 0)
            mf1 = result.get(f"{field}_macro_f1_pct", 0)
            nb  = field_data[field]["n_bewertet"]
            logger.verbose(f"  {field:<12}  acc={acc:.1f}%  macro_f1={mf1:.1f}%  n={nb}")
        for side_key in ("lesions_li", "lesions_re"):
            logger.verbose(
                f"  {side_key:<12}  types: micro_f1={result.get(f'{side_key}_micro_f1_pct')}%  "
                f"exact={result.get(f'{side_key}_exact_match_pct')}%  |  "
                f"count: micro_f1={result.get(f'{side_key}_count_micro_f1_pct')}%  "
                f"exact={result.get(f'{side_key}_count_exact_match_pct')}%"
            )

    return result


def print_mamma_extraction_terminal_report(
    results_csv_path: str, report: dict = None
) -> None:
    r = report or {}
    fields = ["menopause", "birads_li", "birads_re", "acr_li", "acr_re"]
    for field in fields:
        acc = r.get(f"{field}_accuracy_pct", "?")
        mf1 = r.get(f"{field}_macro_f1_pct", "?")
        nb  = "?"
        print(f"  {field:<12}  acc={acc}%  macro_f1={mf1}%")
    for side in ("lesions_li", "lesions_re"):
        print(
            f"  {side:<12}  types: micro_f1={r.get(f'{side}_micro_f1_pct', '?')}%  "
            f"exact={r.get(f'{side}_exact_match_pct', '?')}%  |  "
            f"count: micro_f1={r.get(f'{side}_count_micro_f1_pct', '?')}%  "
            f"exact={r.get(f'{side}_count_exact_match_pct', '?')}%"
        )


# ===========================================================================
# Arm-Röntgen Label Extraction Evaluation
# ===========================================================================

def _binary_prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def _arm_aggregate(label_stats: dict) -> dict:
    """Micro-/Macro-F1 over a set of per-label TP/FP/FN/TN counts.

    Labels with tp+fp+fn == 0 (no positives in GT, none predicted) have an
    undefined F1 and are excluded from the macro average.
    """
    tp = sum(s["tp"] for s in label_stats.values())
    fp = sum(s["fp"] for s in label_stats.values())
    fn = sum(s["fn"] for s in label_stats.values())
    _, _, micro_f1 = _binary_prf(tp, fp, fn)
    defined = [_binary_prf(s["tp"], s["fp"], s["fn"])[2]
               for s in label_stats.values() if s["tp"] + s["fp"] + s["fn"] > 0]
    macro_f1 = sum(defined) / len(defined) if defined else 0.0
    return {"micro_f1": micro_f1, "macro_f1": macro_f1,
            "n_labels": len(label_stats), "n_labels_defined": len(defined)}


def write_arm_extraction_report_jsonl(
    results_csv_path: str,
    out_path: str,
    logger=None,
) -> dict:
    """
    Wertet Arm-Röntgen-Extraktions-CSV aus.

    Labels werden pro Region geführt ("<region> | <label>"), da gleichnamige
    Labels (z.B. "Foreign Bodies") in verschiedenen Regionen verschiedene Aufgaben sind.

    Metriken:
    - F1, Sensitivität, Spezifität pro Label
    - Micro-/Macro-F1 global und pro Region (Macro nur über Labels mit definiertem F1)
    - Accuracy (Anteil korrekter Label-Entscheidungen pro Report, gemittelt)
    - Citation-Match: Anteil der Zitate (finding=true), die wörtlich im Befund stehen;
      geprüft im Task (Spalte citation_check_json), da der Befundtext nicht in der CSV steht
    - 95%-Bootstrap-CI für die Micro-F1 (Report-Ebene)
    """
    df = pd.read_csv(results_csv_path, dtype=str).fillna("")

    n_gesamt      = len(df)
    n_parse_error = int((df["parse_error"].str.lower() == "true").sum())
    has_citation_col = "citation_check_json" in df.columns

    label_stats: dict = {}
    region_of: dict = {}
    per_report_counts: list = []
    per_report_acc: list = []
    citation_ok = citation_total = 0

    for _, row in df.iterrows():
        is_parse_err = row["parse_error"].lower() == "true"
        region = row.get("region", "") or "unknown"

        try:
            gt_labels = _json.loads(row.get("gt_labels_json") or "{}")
        except Exception:
            gt_labels = {}
        try:
            model_labels = _json.loads(row.get("model_labels_json") or "{}") if not is_parse_err else {}
        except Exception:
            model_labels = {}

        row_tp = row_fp = row_fn = row_correct = 0
        for label, gt_val in gt_labels.items():
            try:
                gt_bin = int(gt_val)
            except (ValueError, TypeError):
                gt_bin = 0
            entry = model_labels.get(label, {})
            pred_bin = int(entry.get("finding") is True) if isinstance(entry, dict) else 0

            key = f"{region} | {label}"
            region_of[key] = region
            s = label_stats.setdefault(key, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            if gt_bin and pred_bin:
                s["tp"] += 1; row_tp += 1
            elif pred_bin:
                s["fp"] += 1; row_fp += 1
            elif gt_bin:
                s["fn"] += 1; row_fn += 1
            else:
                s["tn"] += 1
            row_correct += int(gt_bin == pred_bin)

        per_report_counts.append((row_tp, row_fp, row_fn))
        if gt_labels:
            per_report_acc.append(row_correct / len(gt_labels))

        if has_citation_col and not is_parse_err:
            try:
                checks = _json.loads(row.get("citation_check_json") or "{}")
            except Exception:
                checks = {}
            citation_total += len(checks)
            citation_ok += sum(bool(v) for v in checks.values())

    overall = _arm_aggregate(label_stats)
    f1_lo, f1_hi = _bootstrap_micro_f1_ci(per_report_counts)
    acc_mean = sum(per_report_acc) / len(per_report_acc) if per_report_acc else 0.0
    cite_pct = round(citation_ok / citation_total * 100, 2) if citation_total else None

    result: dict = {
        "path":               out_path,
        "n_gesamt":           n_gesamt,
        "n_parse_error":      n_parse_error,
        "micro_f1_pct":       round(overall["micro_f1"] * 100, 2),
        "macro_f1_pct":       round(overall["macro_f1"] * 100, 2),
        "accuracy_pct":       round(acc_mean * 100, 2),
    }
    if cite_pct is not None:
        result["citation_match_pct"] = cite_pct

    regions = sorted(set(region_of.values()))
    region_results = {}
    for region in regions:
        agg = _arm_aggregate({k: v for k, v in label_stats.items() if region_of[k] == region})
        region_results[region] = agg
        result[f"{region}_micro_f1_pct"] = round(agg["micro_f1"] * 100, 2)
        result[f"{region}_macro_f1_pct"] = round(agg["macro_f1"] * 100, 2)

    with open(out_path, "w", encoding="utf-8") as f:
        def _w(obj):
            f.write(_json.dumps(obj, ensure_ascii=False) + "\n")

        _w({"type": "metric", "metric": "n_gesamt",      "value": n_gesamt})
        _w({"type": "metric", "metric": "n_parse_error", "value": n_parse_error})
        _w({"type": "metric", "metric": "micro_f1_pct",  "value": result["micro_f1_pct"],
            "ci_lo": round(f1_lo * 100, 2), "ci_hi": round(f1_hi * 100, 2),
            "note": "CI: bootstrap over reports, micro-F1 recomputed per resample"})
        _w({"type": "metric", "metric": "macro_f1_pct",  "value": result["macro_f1_pct"],
            "n_labels": overall["n_labels"], "n_labels_defined": overall["n_labels_defined"]})
        _w({"type": "metric", "metric": "accuracy_pct",  "value": result["accuracy_pct"]})
        _w({"type": "metric", "metric": "citation_match_pct", "value": cite_pct,
            "n_citations": citation_total})

        for region, agg in region_results.items():
            _w({"type": "region_metric", "region": region,
                "micro_f1_pct": round(agg["micro_f1"] * 100, 2),
                "macro_f1_pct": round(agg["macro_f1"] * 100, 2),
                "n_labels": agg["n_labels"], "n_labels_defined": agg["n_labels_defined"]})

        for key, s in sorted(label_stats.items()):
            tp, fp, fn, tn = s["tp"], s["fp"], s["fn"], s["tn"]
            _, sens, f1 = _binary_prf(tp, fp, fn)
            defined = tp + fp + fn > 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else None
            _w({"type": "label_metric", "region": region_of[key], "label": key.split(" | ", 1)[1],
                "f1_pct": round(f1 * 100, 2) if defined else None,
                "sensitivity_pct": round(sens * 100, 2) if (tp + fn) > 0 else None,
                "specificity_pct": round(spec * 100, 2) if spec is not None else None,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn})

        for _, row in df.iterrows():
            obj = {"type": "item"}
            for col, val in row.to_dict().items():
                obj[col] = _jsonable(val)
            f.write(_json.dumps(obj, ensure_ascii=False) + "\n")

    if logger:
        logger.verbose("\n--- Arm-Röntgen Extraction Evaluation ---")
        logger.verbose(
            f"n={n_gesamt}  parse_err={n_parse_error}  "
            f"micro_f1={result['micro_f1_pct']:.1f}%  macro_f1={result['macro_f1_pct']:.1f}%  "
            f"acc={result['accuracy_pct']:.1f}%  cite_match={cite_pct}"
        )
        for region, agg in region_results.items():
            logger.verbose(f"  {region:<9} micro_f1={agg['micro_f1']*100:.1f}%  macro_f1={agg['macro_f1']*100:.1f}%")

    return result


def print_arm_extraction_terminal_report(
    results_csv_path: str, report: dict = None
) -> None:
    r = report or {}
    cite = r.get("citation_match_pct")
    print(
        f"  Micro-F1: {r.get('micro_f1_pct', '?')}%  "
        f"Macro-F1: {r.get('macro_f1_pct', '?')}%  "
        f"Accuracy: {r.get('accuracy_pct', '?')}%  "
        f"Citation: {f'{cite}%' if cite is not None else 'n/a'}  "
        f"(n={r.get('n_gesamt', '?')}, parse_err={r.get('n_parse_error', '?')})"
    )
    for region in ("clavicle", "elbow", "thumb"):
        if f"{region}_micro_f1_pct" in r:
            print(f"    {region:<9} Micro-F1: {r[f'{region}_micro_f1_pct']}%  "
                  f"Macro-F1: {r[f'{region}_macro_f1_pct']}%")


if __name__ == "__main__":
    main()
