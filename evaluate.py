import argparse
import hashlib
import json
import math
import os
import random
import re
import string
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _read_results_csv(path: str) -> pd.DataFrame:
    """
    Read a results CSV as text. Without dtype=str/keep_default_na=False pandas turns
    model answers such as "None", "NA" or "null" into NaN.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# ===========================================================================
# Confidence intervals (public benchmarks)
# ===========================================================================

_Z95 = 1.959963984540054
N_BOOT = 1000
BOOT_SEED = 42


def wilson_ci(k: float, n: int, z: float = _Z95):
    """Wilson score interval for a proportion k/n. Returns (lo, hi) as fractions."""
    if n <= 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def cluster_bootstrap_ci(values, clusters, n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                         alpha: float = 0.05):
    """
    Percentile bootstrap CI of the mean of *values*, resampling whole clusters
    (e.g. all questions of one image/case) with replacement. The statistic on each
    resample is sum(values) / n_items of the drawn clusters. Returns (lo, hi).
    """
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return None, None
    codes, uniques = pd.factorize(pd.Series([str(c) for c in clusters]))
    k = len(uniques)
    sums = np.bincount(codes, weights=vals, minlength=k)
    counts = np.bincount(codes, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    est = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.percentile(est, 100 * alpha / 2)), float(np.percentile(est, 100 * (1 - alpha / 2)))


def _clusters_of(df: pd.DataFrame):
    """Cluster ids (image / case) if the results CSV has a complete cluster_id column."""
    if "cluster_id" not in df.columns or df.empty:
        return None
    ids = df["cluster_id"].astype(str).str.strip()
    if (ids == "").any() or ids.str.lower().isin(["nan", "none"]).any():
        return None
    return ids.tolist()


def _rate_fields(values, clusters=None) -> dict:
    """
    Report fields for a rate in percent: value, n and a 95% CI.

    Wilson score interval for every proportion. If cluster ids are given, a cluster
    bootstrap (1000 resamples, seed 42, percentile) is the reported CI (ci_method
    "cluster_bootstrap") and Wilson is kept as ci_lo_wilson/ci_hi_wilson. For
    non-binary values (e.g. WBSS) only the cluster/item bootstrap is used.
    """
    vals = np.asarray([float(v) for v in values], dtype=float)
    n = int(len(vals))
    out = {"value": round(float(vals.mean() * 100), 2) if n else None, "n": n}
    if n == 0:
        return out
    binary = bool(np.isin(vals, (0.0, 1.0)).all())
    if binary:
        lo, hi = wilson_ci(float(vals.sum()), n)
        w_lo, w_hi = round(lo * 100, 2), round(hi * 100, 2)
    if clusters is not None:
        b_lo, b_hi = cluster_bootstrap_ci(vals, clusters)
        out.update({"ci_lo": round(b_lo * 100, 2), "ci_hi": round(b_hi * 100, 2),
                    "ci_method": "cluster_bootstrap", "n_clusters": int(len(set(clusters)))})
        if binary:
            out.update({"ci_lo_wilson": w_lo, "ci_hi_wilson": w_hi})
    elif binary:
        out.update({"ci_lo": w_lo, "ci_hi": w_hi, "ci_method": "wilson"})
    else:
        b_lo, b_hi = cluster_bootstrap_ci(vals, list(range(n)))
        out.update({"ci_lo": round(b_lo * 100, 2), "ci_hi": round(b_hi * 100, 2),
                    "ci_method": "bootstrap"})
    return out


# ===========================================================================
# MCQ letter extraction
# ===========================================================================

# Explicit answer statements: "answer is X", "Answer: X", "The correct answer is **X**",
# "best answer: (c)", "Correct Letter: X", "correct option is X", "choice X".
_MARKER_RE = re.compile(
    r"(?i:\b(?:(?:final|correct|best|right)\s+)?(?:answer|choice)\b"
    r"|\bcorrect\s+(?:letter|option)\b)"
    r"[\s*_]*(?i:is|would\s+be|should\s+be|:|=|-)?[\s*_:]*"
    r"(?i:(?:option|letter|choice)\s+)?"
    r"(?P<open>[(\[]*)(?P<letter>[A-Za-z])(?![A-Za-z0-9'’])(?P<close>[*)\]_]*)"
)
# Bold letter: "**B**", "**(B)**", "**B.**"
_BOLD_RE = re.compile(r"\*\*\s*\(?([A-Z])\)?\.?\s*\*\*")
# Leading letter: "B", "B.", "B) ...", "(B) ...", "**B**: ..."
_LEADING_RE = re.compile(r"^[\s*(\[]*([A-Z])[*)\]]*\s*(?:[).:\-]|$)")
# A second option letter right after the first one: "A and B", "A, C", "A/B", "A or B"
_ALSO_RE = re.compile(r"^[*)\]\s]*(?:,|/|&|\+|\band\b|\bor\b)\s*(?:option\s+)?[*(\[]*([A-Za-z])(?![A-Za-z0-9'’])",
                      re.IGNORECASE)
_PRONOUN_I_RE = re.compile(r"^\s*(?:think|believe|would|am|choose|will|'d|'m|’d|’m)\b", re.IGNORECASE)
_NEGATION_RE = re.compile(r"\b(?:not|none|neither|nor|no|cannot|can't)\b", re.IGNORECASE)
_FALLBACK_MAX_CHARS = 40


def _second_letter(text: str, pos: int, first: str, keys: str) -> bool:
    """True if another, different option letter is asserted right after position *pos*."""
    m = _ALSO_RE.match(text[pos:])
    if not m:
        return False
    other = m.group(1)
    if other.islower():
        nxt = text[pos + m.end():pos + m.end() + 1]
        if nxt and nxt not in ".,;:!?)]*\n":
            return False          # "answer is B and a CT ..." – an article, not option a
    return other.upper() in keys and other.upper() != first


def extract_choice(value, valid_keys: str = "ABCDE") -> Optional[str]:
    """
    Parse the chosen option letter from a model reply. Returns None if no single
    answer can be identified.

    Order:
      1. the whole reply is one letter ("B", "(b)", "**C**.");
      2. explicit answer statements ("answer is X", "Answer: X", "Correct Letter: X");
         the LAST one counts, so self-corrections are honoured. Lowercase letters are
         only accepted here ("Answer: d", "best answer: (c)"), not as a word ("the
         answer is a fracture");
      3. bold letters "**X**" (last one counts);
      4. a leading letter ("B. Pneumothorax", "C) ...");
      5. otherwise, for short replies (≤ 40 characters) without a negation, the
         standalone capital option letters, but only if exactly one distinct letter
         occurs ("A and B", "Curve C/D/E", "The answer is not A" → None).
    Two letters asserted together ("answer is A and B", "Both A and C") → None.
    A letter outside *valid_keys* is never returned.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.startswith("Error:"):
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    keys = (valid_keys or "ABCDE").upper()

    bare = text.strip("*()[]{}.:;,!?'\"` \n\t")
    if len(bare) == 1:
        return bare.upper() if bare.upper() in keys else None

    def _scan(regex, allow_lower):
        hits = []
        for m in regex.finditer(text):
            letter = m.group("letter") if "letter" in regex.groupindex else m.group(1)
            if letter.islower():
                if not allow_lower:
                    continue
                nxt = text[m.end():m.end() + 1]
                enclosed = bool(m.group("open")) or bool(m.group("close"))
                if not (enclosed or nxt == "" or nxt in ".,;:!?\n"):
                    continue          # "the answer is a fracture"
            up = letter.upper()
            if up == "I" and _PRONOUN_I_RE.match(text[m.end():]):
                continue
            if up not in keys:
                continue
            hits.append((m.start(), None if _second_letter(text, m.end(), up, keys) else up))
        return hits

    for regex, allow_lower in ((_MARKER_RE, True), (_BOLD_RE, False)):
        hits = _scan(regex, allow_lower)
        if hits:
            return hits[-1][1]

    m = _LEADING_RE.match(text)
    if m and m.group(1) in keys and not _second_letter(text, m.end(1), m.group(1), keys):
        if not (m.group(1) == "I" and _PRONOUN_I_RE.match(text[m.end(1):])):
            return m.group(1)

    # Last resort only for short replies without negation: in longer free text a lone
    # capital letter is not a stated answer ("Comparing the options: **A:** …" cut off,
    # "The answer is not A").
    if len(text) > _FALLBACK_MAX_CHARS or _NEGATION_RE.search(text):
        return None
    candidates = set()
    for m in re.finditer(r"(?<![A-Za-z'’])([A-Z])(?![A-Za-z'’])", text):
        c = m.group(1)
        if c not in keys:
            continue
        if c == "I" and _PRONOUN_I_RE.match(text[m.end():]):
            continue
        if c == "A" and re.match(r"\s+(?!(?:and|or)\b)[a-z]", text[m.end():]):
            continue                  # article at sentence start: "A fracture is seen"
        candidates.add(c)
    return candidates.pop() if len(candidates) == 1 else None


def _row_keys(row) -> str:
    """Valid option keys of a results row (column option_keys, written by tasks/mcq.py)."""
    keys = str(row.get("option_keys") or "").strip().upper()
    return keys or "ABCDE"


def score_results(df: pd.DataFrame) -> pd.DataFrame:
    if "correct_answer" not in df.columns or "model_answer" not in df.columns:
        raise ValueError("CSV must contain columns: correct_answer, model_answer")

    df = df.copy()
    df["correct_answer_norm"] = df.apply(lambda r: extract_choice(r["correct_answer"], _row_keys(r)), axis=1)
    df["model_answer_norm"] = df.apply(lambda r: extract_choice(r["model_answer"], _row_keys(r)), axis=1)
    df["is_correct"] = (df["correct_answer_norm"] == df["model_answer_norm"]) & df["model_answer_norm"].notna()
    return df


def _n_truncated(df: pd.DataFrame) -> Optional[int]:
    """Answers cut off at max_tokens (finish_reason == "length"); None if not recorded."""
    if "finish_reason" not in df.columns:
        return None
    return int((df["finish_reason"].astype(str) == "length").sum())


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


def _write_jsonl_row(f, obj: dict) -> None:
    f.write(json.dumps({k: _jsonable(v) for k, v in obj.items()}, ensure_ascii=False) + "\n")


def write_report_jsonl(
    results_csv_path: str = "results/benchmark_results.csv",
    out_path: str = "results/benchmark_report.jsonl",
    logger=None,
) -> dict:
    """
    Writes a single JSONL file containing:
    - metrics rows (type=metric); accuracy with Wilson 95% CI
    - answer distribution rows (type=answer_distribution)
    - confusion matrix rows (type=confusion, only non-zero cells)
    - per-item scored rows (type=item)
    """
    df = _read_results_csv(results_csv_path)
    scored = score_results(df)
    metrics, dist, conf = compute_reports(scored)

    accuracy_row = metrics.loc[metrics["metric"] == "accuracy_pct", "value"]
    accuracy_pct = float(accuracy_row.iloc[0]) if not accuracy_row.empty else 0.0
    acc_fields = _rate_fields(scored["is_correct"].astype(float), _clusters_of(scored))
    n_trunc = _n_truncated(scored)

    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in metrics.iterrows():
            obj = {"type": "metric", "metric": row["metric"], "value": row["value"]}
            if row["metric"] == "accuracy_pct":
                obj.update(acc_fields)
            _write_jsonl_row(f, obj)
        if n_trunc is not None:
            _write_jsonl_row(f, {"type": "metric", "metric": "n_truncated", "value": n_trunc,
                                 "note": "finish_reason=length (answer cut at max_tokens), still scored"})

        for _, row in dist.iterrows():
            _write_jsonl_row(f, {
                "type": "answer_distribution",
                "answer": row.get("answer"),
                "count": row.get("count"),
                "pct": row.get("pct"),
            })

        for correct in conf.index:
            for model in conf.columns:
                count = int(conf.loc[correct, model])
                if count == 0:
                    continue
                _write_jsonl_row(f, {
                    "type": "confusion",
                    "correct_answer": correct,
                    "model_answer": model,
                    "count": count,
                })

        for _, row in scored.iterrows():
            obj = {"type": "item"}
            obj.update(row.to_dict())
            _write_jsonl_row(f, obj)

    if logger:
        rows = int(metrics.loc[metrics["metric"] == "rows", "value"].iloc[0]) if not metrics.empty else len(df)
        parsed_model = int(metrics.loc[metrics["metric"] == "parsed_model_answer", "value"].iloc[0]) if not metrics.empty else 0
        logger.verbose(f"\n--- MCQ Evaluation ---")
        logger.verbose(f"Total: {rows}  Parsed: {parsed_model}  Accuracy: {accuracy_pct:.2f}% "
                       f"[{acc_fields.get('ci_lo')}, {acc_fields.get('ci_hi')}]"
                       + (f"  truncated: {n_trunc}" if n_trunc else ""))

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

    out = {"accuracy_pct": accuracy_pct, "path": out_path}
    if n_trunc:
        out["n_truncated"] = n_trunc
    return out


def print_terminal_report(results_csv_path: str = "results/benchmark_results.csv") -> None:
    df = _read_results_csv(results_csv_path)
    scored = score_results(df)
    metrics, _, _ = compute_reports(scored)

    accuracy_row = metrics.loc[metrics["metric"] == "accuracy_pct", "value"]
    accuracy_pct = float(accuracy_row.iloc[0]) if not accuracy_row.empty else 0.0
    rows = int(metrics.loc[metrics["metric"] == "rows", "value"].iloc[0]) if not metrics.empty else len(df)
    ci = _rate_fields(scored["is_correct"].astype(float), _clusters_of(scored))

    print(f"  Accuracy: {accuracy_pct:.2f}% [95% CI {ci.get('ci_lo')}–{ci.get('ci_hi')}]  ({rows} questions)")


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


def _exact_match(prediction: str, reference) -> bool:
    """Normalised exact match; *reference* may be a list of accepted alternatives."""
    refs = reference if isinstance(reference, (list, tuple)) else [reference]
    pred = _normalise_text(prediction)
    return any(pred == _normalise_text(r) for r in refs)


def _reference_list(row) -> list:
    """
    All accepted reference answers of a row. VQA-Med-2019 has several alternatives
    for some questions (column reference_answers_json, written by tasks/vqa.py);
    otherwise the single reference_answer.
    """
    raw = row.get("reference_answers_json") if hasattr(row, "get") else None
    if isinstance(raw, str) and raw.strip():
        try:
            refs = [str(r) for r in json.loads(raw) if str(r).strip()]
            if refs:
                return refs
        except Exception:
            pass
    return [str(row.get("reference_answer", "") if hasattr(row, "get") else row)]


def score_vqa_mcq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score MCQ rows in a VQA results CSV.

    Two modes depending on whether reference_answer is a letter or text:
    - Letter reference (e.g. RadImageNet-VQA): extract letter from both sides, compare.
    - Text reference (e.g. RadBench): model picks a letter, look up its text value via
      options_json, compare text to reference case-insensitively.
    Only letters of the question's own options are accepted.
    """
    df = df.copy()

    def _options(row):
        options_raw = row.get("options_json") or "[]"
        try:
            options = json.loads(options_raw) if isinstance(options_raw, str) else (options_raw or [])
        except Exception:
            options = []
        return [o for o in options if isinstance(o, dict) and "key" in o and "value" in o]

    def _valid_keys(row):
        keys = "".join(str(o["key"]).upper() for o in _options(row))
        return keys or "ABCDE"

    def _ref_is_letter(row):
        ref = str(row.get("reference_answer") or "").strip()
        options = _options(row)
        values = {str(o["value"]).strip().lower() for o in options}
        return len(ref) == 1 and ref.upper() in _valid_keys(row) and ref.lower() not in values

    def _score_row(row):
        ref = str(row.get("reference_answer") or "").strip()
        model_letter = row["model_answer_norm"]
        if model_letter is None or (isinstance(model_letter, float) and pd.isna(model_letter)):
            # No letter: accept the option text itself for text references (RadBench)
            return (not _ref_is_letter(row)) and bool(ref) and \
                _normalise_text(row.get("model_answer", "")) == _normalise_text(ref)

        # Case 1: reference is a single letter → classic letter comparison
        if _ref_is_letter(row):
            return model_letter == ref.upper()

        # Case 2: reference is text → map model letter → text via options_json
        letter_map = {str(o["key"]).upper(): str(o["value"]).strip().lower() for o in _options(row)}
        return letter_map.get(model_letter, None) == ref.lower()

    df["correct_answer_norm"] = df.apply(
        lambda r: str(r["reference_answer"]).strip().upper() if _ref_is_letter(r) else str(r["reference_answer"]).strip(),
        axis=1,
    )
    df["model_answer_norm"] = df.apply(lambda r: extract_choice(r["model_answer"], _valid_keys(r)), axis=1)
    df["is_correct"] = df.apply(_score_row, axis=1) if len(df) else pd.Series(dtype=bool)
    return df


def _wbss(prediction: str, reference) -> float:
    """
    Word-Based Semantic Similarity (WBSS) via Wu-Palmer similarity on WordNet.

    WBSS was introduced as a VQA-Med 2018 metric (Hasan et al., ImageCLEF 2018); the
    official VQA-Med 2019 metrics are strict accuracy and BLEU. This is a token-level
    re-implementation (symmetric F-measure of best Wu-Palmer matches), not the
    official scorer, and it gives substantial credit to unrelated answers (see the
    shuffled-reference baseline open_wbss_shuffled_baseline_pct in the report).
    *reference* may be a list of alternatives; the best match counts.
    Requires: nltk + nltk.download('wordnet') + nltk.download('omw-1.4')
    Identical tokens score 1.0 even if they are not in WordNet (e.g. "t2", "cta").
    """
    if isinstance(reference, (list, tuple)):
        return max((_wbss(prediction, r) for r in reference), default=0.0)

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


def _wbss_many(pairs: list) -> list:
    import multiprocessing as _mp
    if len(pairs) < 50:
        return [_wbss(p, r) for p, r in pairs]
    workers = min(_mp.cpu_count(), 8)
    with _mp.Pool(workers) as pool:
        return pool.starmap(_wbss, pairs)


def score_vqa_open(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score open-ended VQA rows: exact match against any reference alternative and WBSS.
    LLM-as-a-Judge is done separately via evaluate_vqa_with_judge().
    """
    df = df.copy()
    refs = [_reference_list(r) for _, r in df.iterrows()]
    answers = df["model_answer"].astype(str).tolist()
    df["wbss"] = _wbss_many(list(zip(answers, refs)))
    df["exact_match"] = [_exact_match(a, r) for a, r in zip(answers, refs)]
    return df


def wbss_shuffled_baseline(df: pd.DataFrame, seed: int = 42) -> float:
    """
    Mean WBSS (fraction) after randomly permuting the references across rows (seed 42):
    the score an answer gets against the reference of an unrelated question.
    """
    refs = [_reference_list(r) for _, r in df.iterrows()]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(refs))
    answers = df["model_answer"].astype(str).tolist()
    vals = _wbss_many([(a, refs[j]) for a, j in zip(answers, perm)])
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# LLM-as-a-Judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT_VERSION = "v2"   # bump when the prompt changes: cached verdicts are reused only for the same version


def parse_judge_reply(raw) -> Optional[int]:
    """Judge verdict 0/1; None if the reply is an error or not a clear verdict."""
    text = str(raw or "")
    if text.startswith("Error:"):
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.fullmatch(r"[\s*'\"`]*([01])[\s*'\"`.]*", text)
    return int(m.group(1)) if m else None


def judge_prompt(question: str, references: list, model_answer: str) -> str:
    refs = " | ".join(str(r) for r in references)
    return (
        "You are a medical expert judge evaluating a model's answer to a radiology question.\n\n"
        f"Question: {question}\n"
        f"Reference answer(s): {refs}\n"
        f"Model answer: {model_answer}\n\n"
        "Is the model answer correct? Rules:\n"
        "- Correct if it means the same as a reference answer at the level of detail the question asks for "
        "(synonyms, abbreviations and different wording are fine).\n"
        "- A more specific answer that is still correct counts as correct.\n"
        "- Wrong if it contradicts the reference, names a different finding, or adds findings that are wrong.\n"
        "Reply with exactly '1' (correct) or '0' (incorrect). No other text."
    )


def _judge_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9.\-]+", "_", str(name or "unknown")).strip("_") or "unknown"


def _judge_key(model_answer, references) -> str:
    payload = json.dumps([str(model_answer), [str(r) for r in references]], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _judge_model_name(client) -> str:
    return str(getattr(client, "model", None) or "unknown")


def evaluate_vqa_with_judge(
    df: pd.DataFrame,
    client,
    cache_path: Optional[str] = None,
    workers: int = 8,
) -> pd.DataFrame:
    """
    LLM-as-a-Judge evaluation for open-ended VQA rows.

    Binary correct/incorrect verdict (as in RadImageNet-VQA, Butsanets et al., 2025;
    LLM-as-a-judge: Zheng et al., 2023). The judge sees the question, all reference
    answers and the model prediction, plus a short rubric, and returns 1 or 0.

    Verdicts are cached in *cache_path* (CSV: id, key, judge_model, prompt_version,
    judge_raw). A cached verdict is reused only if item id, sha1 of (model answer,
    references), judge model and prompt version all match. Only parseable verdicts are
    reused; judge errors and unparsed replies are asked again on the next evaluation.

    Adds columns:
      judge_raw     – raw judge reply
      judge_status  – ok | unparsed | judge_error | model_error (model answer was an API error, not judged)
      judge_correct – 1/0; unparsed replies, judge errors and model errors count as 0
    core.client.ServerUnavailableError from the judge client propagates.
    """
    import csv as _csv
    from concurrent.futures import ThreadPoolExecutor, as_completed

    df = df.copy()
    judge_model = _judge_model_name(client)
    ids = df["id"].astype(str).tolist()
    refs = [_reference_list(r) for _, r in df.iterrows()]
    answers = df["model_answer"].astype(str).tolist()
    keys = [_judge_key(a, r) for a, r in zip(answers, refs)]

    fields = ["id", "key", "judge_model", "prompt_version", "judge_raw"]
    cache: dict = {}
    if cache_path and os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        cached = _read_results_csv(cache_path)
        for _, r in cached.iterrows():
            if (r.get("judge_model") == judge_model and r.get("prompt_version") == JUDGE_PROMPT_VERSION
                    and parse_judge_reply(r.get("judge_raw")) is not None):
                cache[(r["id"], r["key"])] = r["judge_raw"]

    raw_by_pos: dict = {}
    todo = []
    for pos, (i, k, a) in enumerate(zip(ids, keys, answers)):
        if a.startswith("Error:"):
            continue
        if (i, k) in cache:
            raw_by_pos[pos] = cache[(i, k)]
        else:
            todo.append(pos)
    if todo:
        print(f"  LLM-Judge ({judge_model}): {len(todo)} answers to judge "
              f"({len(raw_by_pos)} cached, {workers} workers)...")

    cache_file = None
    writer = None
    if cache_path and todo:
        new_file = not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 0)
        cache_file = open(cache_path, "a", newline="", encoding="utf-8")
        writer = _csv.DictWriter(cache_file, fieldnames=fields)
        if new_file:
            writer.writeheader()

    questions = df["question"].astype(str).tolist() if "question" in df.columns else [""] * len(df)

    def _judge(pos):
        return client.ask_question(judge_prompt(questions[pos], refs[pos], answers[pos]))

    try:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {pool.submit(_judge, pos): pos for pos in todo}
            try:
                for n_done, fut in enumerate(as_completed(futures), start=1):
                    pos = futures[fut]
                    raw = fut.result()          # ServerUnavailableError propagates
                    raw_by_pos[pos] = raw
                    if writer and not str(raw).startswith("Error:"):
                        writer.writerow({"id": ids[pos], "key": keys[pos], "judge_model": judge_model,
                                         "prompt_version": JUDGE_PROMPT_VERSION, "judge_raw": raw})
                        cache_file.flush()
                    if n_done % 100 == 0:
                        print(f"    judged {n_done}/{len(todo)}")
            except BaseException:
                for f in futures:
                    f.cancel()
                raise
    finally:
        if cache_file:
            cache_file.close()

    raws, statuses, verdicts = [], [], []
    for pos, a in enumerate(answers):
        if a.startswith("Error:"):
            raws.append(""); statuses.append("model_error"); verdicts.append(0)
            continue
        raw = str(raw_by_pos.get(pos, ""))
        v = parse_judge_reply(raw)
        raws.append(raw)
        if v is not None:
            statuses.append("ok"); verdicts.append(v)
        elif raw.startswith("Error:") or not raw:
            statuses.append("judge_error"); verdicts.append(0)
        else:
            statuses.append("unparsed"); verdicts.append(0)
    df["judge_raw"] = raws
    df["judge_status"] = statuses
    df["judge_correct"] = verdicts
    df["judge_model"] = judge_model
    return df


def _judge_summary(scored: pd.DataFrame) -> dict:
    """Judge accuracy over all rows (unparsed/errors = wrong) plus the counts."""
    status = scored["judge_status"] if "judge_status" in scored.columns else pd.Series(["ok"] * len(scored))
    out = {
        "n_judged": int((status == "ok").sum()),
        "n_judge_unparsed": int((status == "unparsed").sum()),
        "n_judge_errors": int((status == "judge_error").sum()),
    }
    if len(scored):
        out["judge_accuracy_pct"] = round(float(pd.to_numeric(scored["judge_correct"]).fillna(0).mean() * 100), 2)
    if out["n_judge_unparsed"] or out["n_judge_errors"]:
        print(f"  WARNING: {out['n_judge_unparsed']} unparsed judge replies and "
              f"{out['n_judge_errors']} judge errors (counted as wrong in LLM-Judge accuracy)")
    return out


def _yes_no_token(text) -> Optional[str]:
    words = _normalise_text(str(text or "")).split()
    return words[0] if words and words[0] in ("yes", "no") else None


def _judge_cache_path(results_csv_path: str, judge_model: str = "unknown") -> str:
    """{bench}_judge_cache_{judge-model-slug}.csv next to the results CSV."""
    base = results_csv_path[:-len("_results.csv")] if results_csv_path.endswith("_results.csv") else results_csv_path
    return f"{base}_judge_cache_{_judge_slug(judge_model)}.csv"


def _judge_workers(config: Optional[dict], default: int = 8) -> int:
    """Judge concurrency: judge.concurrency, else benchmark_settings.concurrency, else 8."""
    if not config:
        return default
    for section in ("judge", "benchmark_settings"):
        val = (config.get(section) or {}).get("concurrency")
        if val:
            return max(1, int(val))
    return default


def _category_values(df: pd.DataFrame) -> list:
    if "category" not in df.columns:
        return []
    cats = df["category"].astype(str).str.strip()
    return sorted(c for c in cats.unique() if c)


def _subset_rows(df: pd.DataFrame, subset: str, columns: dict) -> list:
    """
    Metric rows for a subset and, if a category column exists, per category
    (subset "open:category=plane"). *columns* maps metric name → per-row value column.
    """
    rows = []
    parts = [(subset, df)]
    for cat in _category_values(df):
        parts.append((f"{subset}:category={cat}", df[df["category"].astype(str).str.strip() == cat]))
    for name, part in parts:
        clusters = _clusters_of(part)
        for metric, col in columns.items():
            if col not in part.columns:
                continue
            vals = pd.to_numeric(part[col].astype(float), errors="coerce").fillna(0)
            rows.append({"type": "metric", "subset": name, "metric": metric, **_rate_fields(vals, clusters)})
    return rows


def write_vqa_report_jsonl(
    results_csv_path: str,
    out_path: str,
    client=None,
    run_judge: bool = False,
    logger=None,
    config: Optional[dict] = None,
    judge_workers: Optional[int] = None,
) -> dict:
    """
    Evaluate a VQA results CSV and write a JSONL report.

    MCQ rows    → letter accuracy (only the question's option letters are accepted).
    Yes/No rows → first word yes/no.
    Open rows   → exact match (any reference alternative), WBSS (+ shuffled-reference
                  baseline) and optionally LLM-as-a-Judge.
    Every accuracy gets a 95% CI (Wilson; cluster bootstrap over images/cases if the
    CSV has cluster_id). If the CSV has a category column, metrics are also reported
    per category (subset "<type>:category=<value>").
    """
    df = _read_results_csv(results_csv_path)

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
    workers = judge_workers or _judge_workers(config)

    # Judge before opening the report file, so an interrupted judge run never
    # leaves an empty report behind (verdicts are cached next to the results CSV).
    scored_open = None
    shuffled_wbss = None
    judge_model = None
    if not open_df.empty:
        scored_open = score_vqa_open(open_df)
        shuffled_wbss = wbss_shuffled_baseline(open_df)
        if run_judge and client is not None:
            judge_model = _judge_model_name(client)
            scored_open = evaluate_vqa_with_judge(
                scored_open, client,
                cache_path=_judge_cache_path(results_csv_path, judge_model),
                workers=workers,
            )

    scored_mcq = score_vqa_mcq(mcq_df) if not mcq_df.empty else None
    scored_yn = None
    if not yes_no_df.empty:
        # RadImageNet-VQA "checks for the expected token": compare the first word.
        scored_yn = yes_no_df.copy()
        scored_yn["is_correct"] = scored_yn.apply(
            lambda r: _yes_no_token(r["model_answer"]) == _yes_no_token(r["reference_answer"])
            and _yes_no_token(r["reference_answer"]) is not None,
            axis=1,
        )

    with open(out_path, "w", encoding="utf-8") as f:
        def _w(obj):
            _write_jsonl_row(f, obj)

        def _common(subset, part):
            _w({"type": "metric", "subset": subset, "metric": "rows", "value": len(part)})
            nt = _n_truncated(part)
            if nt is not None:
                _w({"type": "metric", "subset": subset, "metric": "n_truncated", "value": nt,
                    "note": "finish_reason=length (answer cut at max_tokens), still scored"})
                if nt:
                    results[f"{subset}_n_truncated"] = nt

        def _items(subset, part):
            for _, row in part.iterrows():
                obj = {"type": "item", "subset": subset}
                obj.update(row.to_dict())
                _w(obj)

        # --- MCQ sub-results ---
        if scored_mcq is not None:
            scored_mcq["is_correct"] = scored_mcq["is_correct"].astype(bool)
            rows = _subset_rows(scored_mcq, "mcq", {"accuracy_pct": "is_correct"})
            results["mcq_accuracy_pct"] = rows[0]["value"]
            results["mcq_rows"] = len(scored_mcq)
            n_unparsed = int(scored_mcq["model_answer_norm"].isna().sum())
            for r in rows:
                _w(r)
            _common("mcq", scored_mcq)
            _w({"type": "metric", "subset": "mcq", "metric": "n_unparsed", "value": n_unparsed,
                "note": "no single option letter could be extracted (counted as wrong)"})
            _items("mcq", scored_mcq)

        # --- Yes/No (closed-ended) sub-results ---
        if scored_yn is not None:
            rows = _subset_rows(scored_yn, "yes_no", {"accuracy_pct": "is_correct"})
            rows[0]["note"] = "first token yes/no, RadImageNet-VQA closed-ended task"
            results["yes_no_accuracy_pct"] = rows[0]["value"]
            results["yes_no_rows"] = len(scored_yn)
            for r in rows:
                _w(r)
            _common("yes_no", scored_yn)
            _items("yes_no", scored_yn)

        # --- Open-ended sub-results ---
        if scored_open is not None:
            cols = {"exact_match_pct": "exact_match", "wbss_pct": "wbss"}
            if "judge_correct" in scored_open.columns:
                cols["llm_judge_accuracy_pct"] = "judge_correct"
            rows = _subset_rows(scored_open, "open", cols)
            notes = {
                "exact_match_pct": "normalised exact match against any reference alternative "
                                   "(VQA-Med-2019 official metric is strict accuracy)",
                "wbss_pct": "Wu-Palmer word similarity (WBSS, VQA-Med 2018 metric); secondary, "
                            "compare with wbss_shuffled_baseline_pct",
                "llm_judge_accuracy_pct": "binary correct/incorrect over all rows; unparsed replies "
                                          "and judge errors count as wrong",
            }
            js = _judge_summary(scored_open) if "judge_correct" in scored_open.columns else None
            for r in rows:
                if r["subset"] == "open":
                    r["note"] = notes[r["metric"]]
                if r["metric"] == "llm_judge_accuracy_pct":
                    r["judge_model"] = judge_model
                    if r["subset"] == "open":
                        r.update({k: js[k] for k in ("n_judged", "n_judge_unparsed", "n_judge_errors")})
                _w(r)
            top = {r["metric"]: r["value"] for r in rows if r["subset"] == "open"}
            results["open_exact_match_pct"] = top["exact_match_pct"]
            results["open_wbss_pct"] = top["wbss_pct"]
            results["open_rows"] = len(scored_open)
            base = round(shuffled_wbss * 100, 2)
            results["open_wbss_shuffled_baseline_pct"] = base
            _w({"type": "metric", "subset": "open", "metric": "wbss_shuffled_baseline_pct", "value": base,
                "note": "WBSS with references randomly permuted across questions (seed 42): "
                        "the score of an answer to a different question"})
            if js is not None:
                results["open_judge_accuracy_pct"] = top["llm_judge_accuracy_pct"]
                results["open_n_judge_unparsed"] = js["n_judge_unparsed"]
                results["open_n_judge_errors"] = js["n_judge_errors"]
                results["judge_model"] = judge_model
                _w({"type": "info", "key": "judge_model", "value": judge_model,
                    "prompt_version": JUDGE_PROMPT_VERSION})
            _common("open", scored_open)
            _items("open", scored_open)

    if logger:
        logger.verbose("\n--- VQA Evaluation ---")
        if scored_mcq is not None:
            logger.verbose(f"MCQ: {results.get('mcq_rows', 0)} questions  Accuracy: {results.get('mcq_accuracy_pct', 0):.2f}%")
            wrong_mcq = scored_mcq[~scored_mcq["is_correct"]].head(10)
            if not wrong_mcq.empty:
                logger.verbose(f"  Wrong MCQ examples (first {len(wrong_mcq)}):")
                for _, row in wrong_mcq.iterrows():
                    logger.verbose(
                        f"    [{row.get('id')}] correct={row['correct_answer_norm']}  "
                        f"model={row['model_answer_norm']}  raw={str(row.get('model_answer',''))[:30]!r}"
                    )
        if scored_yn is not None:
            logger.verbose(f"Yes/No: {results.get('yes_no_rows', 0)} questions  Accuracy: {results.get('yes_no_accuracy_pct', 0):.2f}%")
        if scored_open is not None:
            logger.verbose(
                f"Open: {results.get('open_rows', 0)} questions  Exact: {results.get('open_exact_match_pct', 0):.2f}%"
                f"  WBSS: {results.get('open_wbss_pct', 0):.2f}% (shuffled baseline "
                f"{results.get('open_wbss_shuffled_baseline_pct', 0):.2f}%)"
                + (f"  LLM-Judge: {results.get('open_judge_accuracy_pct', 0):.2f}%" if "open_judge_accuracy_pct" in results else "")
            )
            # Bottom-20 open questions by WBSS
            bottom = scored_open.nsmallest(20, "wbss")
            logger.verbose(f"  Bottom {len(bottom)} open answers by WBSS:")
            for _, row in bottom.iterrows():
                q_short = str(row.get("question", ""))[:60]
                ref_short = " | ".join(_reference_list(row))[:40]
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
        judge_str = ""
        if "open_judge_accuracy_pct" in r:
            judge_str = f"  LLM-Judge: {r['open_judge_accuracy_pct']:.2f}% ({r.get('judge_model', '?')})"
        bad = (r.get("open_n_judge_unparsed") or 0) + (r.get("open_n_judge_errors") or 0)
        bad_str = f" ({bad} unparsed/errors counted wrong)" if bad else ""
        parts.append(
            f"Open ({r.get('open_rows', '?')} questions): Exact {r.get('open_exact_match_pct', 0):.2f}%  "
            f"WBSS {r['open_wbss_pct']:.2f}% (shuffled {r.get('open_wbss_shuffled_baseline_pct', 0):.2f}%)"
            f"{judge_str}{bad_str}"
        )

    for p in parts:
        print(f"  {p}")


# ===========================================================================
# Mamma-MRT Label Extraction Evaluation
# ===========================================================================


_MAMMA_NORM_PATH = Path("config/mamma_normalization.yaml")
_MAMMA_FIELDS = ["menopause", "birads_li", "birads_re", "acr_li", "acr_re"]

# Sensitivity analyses (always computed next to the configured primary definition).
# Naming scheme: same metric as the primary one with a variant tag before "_pct",
#   JSONL row : {"type": "metric", "field": f, "metric": "accuracy_<tag>_pct", "variant": <variant>, ...}
#   summary   : "<field>_accuracy_<tag>_pct", e.g. "birads_li_accuracy_birads6keep_pct"
# so the flattened summary never collides with the primary keys ("<field>_accuracy_pct").
# The variant is the *other* option of the configured setting.
_BIRADS6_VARIANTS = {  # configured primary -> (alternative option, tag, variant name)
    "map_to_5": ("keep", "birads6keep", "birads6_keep"),
    "keep": ("map_to_5", "birads6mapto5", "birads6_map_to_5"),
}
_GT_EMPTY_VARIANTS = {
    "ignore": (True, "gtemptyfp", "gt_empty_fp"),
    "fp": (False, "gtemptyignore", "gt_empty_ignore"),
}

_MACRO_F1_NOTE = (
    "macro-F1 over GT classes; a missing model value counts as FN of the GT class but as "
    "FP of no class, so with coverage < 100% macro-F1 can exceed accuracy (accuracy counts "
    "missing values as wrong)"
)
_ACR_SIDE_NOTE = (
    "BPE is usually one value per exam: li and re rows are largely the same decision "
    "counted twice; see acr_exam_accuracy_pct for the exam-level view"
)


@lru_cache(maxsize=1)
def _load_mamma_norm() -> dict:
    """Normalisation mapping from config/mamma_normalization.yaml (cached); {} if the file is missing."""
    if not _MAMMA_NORM_PATH.exists():
        return {}
    import yaml
    with open(_MAMMA_NORM_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_normalizer(mapping: dict) -> dict:
    """Inverted mapping: variant.lower().strip() → canonical form."""
    inv: dict = {}
    for canonical, variants in (mapping or {}).items():
        key = str(canonical).lower().strip()
        inv[key] = str(canonical)
        for v in (variants or []):
            inv[str(v).lower().strip()] = str(canonical)
    return inv


def _normalize_val(value, normalizer: dict):
    """Canonical form of a value (lowercased value if it is not in the mapping); None if empty."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s in ("nan", "None", "?"):
        return None
    return normalizer.get(s.lower(), s.lower())


def _normalize_birads(value, normalizer: dict, birads6_handling: str = "map_to_5"):
    """Canonical BI-RADS category; takes the leading digit of texts such as '6 nachgewiesene ...'."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s in ("nan", "None", "?", "ERROR"):
        return None

    # Mapping from the normalisation YAML first
    normed = normalizer.get(s.lower())

    # Fallback: leading digit (e.g. "4 Suspekt..." → "4")
    if normed is None:
        m = re.search(r"(?<!\d)([1-6])(?!\d)", s)
        if m:
            normed = m.group(1)
        else:
            # Plain number?
            try:
                normed = str(int(float(s)))
            except (ValueError, TypeError):
                normed = None

    if normed == "6" and birads6_handling == "map_to_5":
        normed = "5"
    return normed


def _normalize_acr(value, normalizer: dict, acr_range: str = "min"):
    """
    Canonical ACR/BPE value. Ranges such as '1 bis 2' are resolved by the acr_range
    option. Returns (canonical value, is_range_error).
    """
    if value is None:
        return None, False
    s = str(value).strip()
    if not s or s in ("nan", "None", "?", "ERROR"):
        return None, False

    # Range, e.g. "1 bis 2", "2-3", "1 to 2"
    m_range = re.search(r"(?<!\d)([1-4])\s*(?:bis|to|-|–)\s*([1-4])(?!\d)", s, re.IGNORECASE)
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
        # Leading digit
        m = re.search(r"(?<!\d)([1-4])(?!\d)", s)
        if m:
            normed = m.group(1)
        else:
            try:
                v = int(float(s))
                normed = str(v) if 1 <= v <= 4 else None
            except (ValueError, TypeError):
                normed = None
    return normed, False


def _multiset_prf(gt_list: list, pred_list: list) -> tuple:
    """Multiset TP/FP/FN of two lesion lists."""
    gt_c   = Counter(gt_list)
    pred_c = Counter(pred_list)
    tp = sum(min(gt_c[k], pred_c.get(k, 0)) for k in gt_c)
    fp = sum(max(0, pred_c[k] - gt_c.get(k, 0)) for k in pred_c)
    fn = sum(max(0, gt_c[k] - pred_c.get(k, 0)) for k in gt_c)
    return tp, fp, fn


def _bootstrap_ci(values: list, n_boot: int = 1000, alpha: float = 0.05) -> tuple:
    """95% bootstrap CI of a mean over reports (seed 42). Returns (point, lo, hi)."""
    if not values:
        return 0.0, 0.0, 0.0
    n = len(values)
    if n == 1:
        v = float(values[0])
        return v, v, v
    rng = random.Random(42)
    boot = sorted(
        sum(values[rng.randint(0, n - 1)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    lo = boot[int(n_boot * alpha / 2)]
    hi = boot[int(n_boot * (1 - alpha / 2)) - 1]
    return sum(values) / n, lo, hi


def _bootstrap_micro_prf_ci(counts: list, n_boot: int = 1000, alpha: float = 0.05) -> dict:
    """
    95% CIs for micro precision / recall / F1: resample reports (seed 42), sum their
    (tp, fp, fn), recompute.
    Returns {"precision": (lo, hi), "recall": (lo, hi), "f1": (lo, hi)} as fractions.
    """
    if not counts:
        return {"precision": (0.0, 0.0), "recall": (0.0, 0.0), "f1": (0.0, 0.0)}
    n = len(counts)
    rng = random.Random(42)
    ps, rs, fs = [], [], []
    for _ in range(n_boot):
        sample = [counts[rng.randint(0, n - 1)] for _ in range(n)]
        tp = sum(c[0] for c in sample)
        fp = sum(c[1] for c in sample)
        fn = sum(c[2] for c in sample)
        ps.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        rs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        denom = 2 * tp + fp + fn
        fs.append(2 * tp / denom if denom > 0 else 0.0)
    lo_i, hi_i = int(n_boot * alpha / 2), int(n_boot * (1 - alpha / 2)) - 1
    out = {}
    for name, vals in (("precision", ps), ("recall", rs), ("f1", fs)):
        vals.sort()
        out[name] = (vals[lo_i], vals[hi_i])
    return out


def _categorical_metrics(y_true: list, y_pred: list) -> dict:
    """Accuracy, macro-F1 and confusion matrix of two parallel lists."""
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
    """JSON list string → list of canonical lesion types."""
    try:
        items = json.loads(raw_json) if isinstance(raw_json, str) else (raw_json or [])
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


def _truncation_counts(df: pd.DataFrame):
    """(n_truncated, n_parse_error_truncated) from finish_reason; (None, None) if not recorded."""
    if "finish_reason" not in df.columns:
        return None, None
    trunc = df["finish_reason"].str.lower() == "length"
    perr = df["parse_error"].str.lower() == "true"
    return int(trunc.sum()), int((trunc & perr).sum())


def _mamma_norms() -> dict:
    norm = _load_mamma_norm()
    return {
        "menopause": _build_normalizer(norm.get("menopause", {})),
        "birads":    _build_normalizer(norm.get("birads", {})),
        "acr":       _build_normalizer(norm.get("acr", {})),
        "lesion":    _build_normalizer(norm.get("lesion_types", {})),
    }


def _mamma_norm_field(field: str, raw, norms: dict, acr_range: str, birads6_handling: str):
    prefix = field.split("_")[0]
    if prefix == "menopause":
        return _normalize_val(raw, norms["menopause"])
    if prefix == "birads":
        return _normalize_birads(raw, norms["birads"], birads6_handling)
    val, rng_err = _normalize_acr(raw, norms["acr"], acr_range)
    return None if rng_err else val


def _score_mamma(df: pd.DataFrame, norms: dict, acr_range: str, birads6_handling: str,
                 gt_empty_fp: bool):
    """Collects per-field and per-lesion-side scoring data for one definition."""
    field_data: dict = {
        f: {"y_true": [], "y_pred": [], "n_scored": 0,
            "n_gt_empty_model_present": 0, "n_both_empty": 0, "n_model_missing": 0,
            "per_item_correct": []}
        for f in _MAMMA_FIELDS
    }
    # Two lesion views per side:
    #   "type"  (main metric): which lesion types occur on the side (set comparison)
    #   "count" (secondary):   one entry per lesion (multiset); GT has one row per
    #                          lesion, so multifocal findings count several times
    lesion_sides = {
        (side, mode): {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "n": 0, "ignored": 0,
                       "ignored_model_present": 0, "ignored_model_entries": 0,
                       "per_item_counts": []}
        for side in ("lesions_li", "lesions_re") for mode in ("type", "count")
    }

    for _, row in df.iterrows():
        is_parse_err = row["parse_error"].lower() == "true"

        for field in _MAMMA_FIELDS:
            fd = field_data[field]
            gt_n = _mamma_norm_field(field, row.get(f"gt_{field}", ""), norms, acr_range, birads6_handling)
            ext_n = None if is_parse_err else _mamma_norm_field(
                field, row.get(f"model_{field}", ""), norms, acr_range, birads6_handling)

            if gt_n is None and ext_n is None:
                fd["n_both_empty"] += 1
            elif gt_n is None:
                fd["n_gt_empty_model_present"] += 1
                if gt_empty_fp:
                    fd["y_true"].append("__empty__")
                    fd["y_pred"].append(ext_n)
                    fd["n_scored"] += 1
                    fd["per_item_correct"].append(0)
            elif ext_n is None:
                fd["y_true"].append(gt_n)
                fd["y_pred"].append("__missing__")
                fd["n_scored"] += 1
                fd["n_model_missing"] += 1
                fd["per_item_correct"].append(0)
            else:
                fd["y_true"].append(gt_n)
                fd["y_pred"].append(ext_n)
                fd["n_scored"] += 1
                fd["per_item_correct"].append(int(gt_n == ext_n))

        for side_key in ("lesions_li", "lesions_re"):
            gt_list = _normalize_lesion_list(row.get(f"gt_{side_key}", "[]"), norms["lesion"])
            ext_list = ([] if is_parse_err else
                        _normalize_lesion_list(row.get(f"model_{side_key}", "[]"), norms["lesion"]))

            for mode, (g, p) in (
                ("type",  (sorted(set(gt_list)), sorted(set(ext_list)))),
                ("count", (gt_list, ext_list)),
            ):
                sd = lesion_sides[(side_key, mode)]
                # The GT lesion table only lists lesions with histology/follow-up, so a
                # side without GT lesions is not scored (same rule as for empty GT fields)
                if not g and not gt_empty_fp:
                    sd["ignored"] += 1
                    if p:
                        sd["ignored_model_present"] += 1
                        sd["ignored_model_entries"] += len(p)
                    continue
                tp, fp, fn = _multiset_prf(g, p)
                sd["tp"] += tp
                sd["fp"] += fp
                sd["fn"] += fn
                sd["n"]  += 1
                sd["per_item_counts"].append((tp, fp, fn))
                sd["exact"] += int(Counter(g) == Counter(p))
    return field_data, lesion_sides


def _mamma_field_summary(fd: dict) -> dict:
    yt, yp = fd["y_true"], fd["y_pred"]
    cm_res = _categorical_metrics(yt, yp)
    scored = bool(yt)
    _, acc_lo, acc_hi = _bootstrap_ci(fd["per_item_correct"])
    # accuracy among the items where the model gave a value (missing values left out):
    # separates "wrong value" from "no value" (e.g. menopause not stated in the report)
    answered = [(t, p) for t, p in zip(yt, yp) if p != "__missing__" and t != "__empty__"]
    acc_answered = (round(sum(t == p for t, p in answered) / len(answered) * 100, 2)
                    if answered else None)

    def _pct(x):
        # Nothing to score → undefined (None), not 0%
        return round(x * 100, 2) if scored else None

    return {
        "accuracy": _pct(cm_res["accuracy"]), "acc_lo": _pct(acc_lo), "acc_hi": _pct(acc_hi),
        "macro_f1": _pct(cm_res["macro_f1"]),
        "coverage": _pct((fd["n_scored"] - fd["n_model_missing"]) / fd["n_scored"]) if scored else None,
        "accuracy_answered": acc_answered, "n_answered": len(answered),
        "confusion": cm_res["confusion"],
    }


def _mamma_lesion_summary(sd: dict) -> dict:
    tp, fp, fn = sd["tp"], sd["fp"], sd["fn"]
    micro_p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    defined = (tp + fp + fn) > 0
    cis = _bootstrap_micro_prf_ci(sd["per_item_counts"])

    def _pct(x):
        return round(x * 100, 2) if defined else None

    return {
        "defined": defined,
        "precision": _pct(micro_p), "precision_ci": tuple(_pct(v) for v in cis["precision"]),
        "recall": _pct(micro_r), "recall_ci": tuple(_pct(v) for v in cis["recall"]),
        "f1": _pct(micro_f1), "f1_ci": tuple(_pct(v) for v in cis["f1"]),
        "f1_raw": micro_f1,
        "exact": round(sd["exact"] / sd["n"] * 100, 2) if sd["n"] > 0 else None,
    }


def _acr_exam_level(df: pd.DataFrame, norms: dict, acr_range: str) -> dict:
    """
    Exam-level BPE accuracy: GT exam value = GT li if GT li == GT re (or the only GT side);
    exams with GT li != re are excluded and counted. Correct iff every non-empty model
    side equals the GT exam value (a missing model value counts as wrong).
    """
    correct: list = []
    n_gt_differs = n_one_side = n_model_differs = 0
    for _, row in df.iterrows():
        is_parse_err = row["parse_error"].lower() == "true"
        g_li = _mamma_norm_field("acr_li", row.get("gt_acr_li", ""), norms, acr_range, "map_to_5")
        g_re = _mamma_norm_field("acr_re", row.get("gt_acr_re", ""), norms, acr_range, "map_to_5")
        if g_li is None and g_re is None:
            continue
        if g_li is not None and g_re is not None and g_li != g_re:
            n_gt_differs += 1
            continue
        if g_li is None or g_re is None:
            n_one_side += 1
        gt_val = g_li if g_li is not None else g_re
        m_vals = [] if is_parse_err else [
            v for v in (
                _mamma_norm_field("acr_li", row.get("model_acr_li", ""), norms, acr_range, "map_to_5"),
                _mamma_norm_field("acr_re", row.get("model_acr_re", ""), norms, acr_range, "map_to_5"),
            ) if v is not None
        ]
        if len(set(m_vals)) > 1:
            n_model_differs += 1
        correct.append(int(bool(m_vals) and set(m_vals) == {gt_val}))
    point, lo, hi = _bootstrap_ci(correct)
    scored = bool(correct)
    return {
        "accuracy": round(point * 100, 2) if scored else None,
        "ci_lo": round(lo * 100, 2) if scored else None,
        "ci_hi": round(hi * 100, 2) if scored else None,
        "n_exams": len(correct), "n_gt_differs": n_gt_differs,
        "n_one_side": n_one_side, "n_model_differs": n_model_differs,
    }


def _menopause_in_report_level(df: pd.DataFrame, norms: dict, acr_range: str) -> dict:
    """
    Menopause accuracy restricted to exams whose report text mentions the status
    (column menopause_in_report, written by the task). The annotation often takes the
    status from other sources, so the plain accuracy counts "not in the report" as a
    model error; this view measures the extraction itself. A missing model value or a
    parse error counts as wrong. Returns accuracy None if the column is absent.
    """
    if "menopause_in_report" not in df.columns:
        return {"accuracy": None}
    flags = df["menopause_in_report"].astype(str).str.lower()
    if not flags.isin(["true", "false"]).any():
        return {"accuracy": None}
    correct: list = []
    n_not_in_report = 0
    for (_, row), flag in zip(df.iterrows(), flags):
        gt = _mamma_norm_field("menopause", row.get("gt_menopause", ""), norms, acr_range, "map_to_5")
        if gt is None or flag not in ("true", "false"):
            continue
        if flag == "false":
            n_not_in_report += 1
            continue
        pred = (None if row["parse_error"].lower() == "true" else
                _mamma_norm_field("menopause", row.get("model_menopause", ""), norms, acr_range, "map_to_5"))
        correct.append(int(pred == gt))
    point, lo, hi = _bootstrap_ci(correct)
    scored = bool(correct)
    return {
        "accuracy": round(point * 100, 2) if scored else None,
        "ci_lo": round(lo * 100, 2) if scored else None,
        "ci_hi": round(hi * 100, 2) if scored else None,
        "n": len(correct), "n_not_in_report": n_not_in_report,
    }


def write_mamma_extraction_report_jsonl(
    results_csv_path: str,
    out_path: str,
    config: dict = None,
    logger=None,
) -> dict:
    """
    Evaluate a Mamma-MRT extraction results CSV.

    Config options (task_settings.label_extraction_mamma):
        acr_range           : "min" (default) | "max" | "error"
        birads6_handling    : "map_to_5" (default) | "keep"
        gt_empty_ext_present: "ignore" (default) | "fp"  – applies to fields and lesion sides

    Primary metrics follow the configured definition. Sensitivity analyses are always
    added for the respective other option of birads6_handling (BI-RADS fields) and of
    gt_empty_ext_present (all fields + lesion sides); naming: see _BIRADS6_VARIANTS.

    Metrics per categorical field: accuracy (+CI), macro-F1, coverage, confusion matrix.
    BPE also at exam level (acr_exam_accuracy_pct). Menopause also on the exams whose report
    text mentions the status (menopause_accuracy_in_report_pct; needs column menopause_in_report).
    Lesions per side: main metric = set of lesion types (which types occur), plus a
    count view (multiset, one entry per lesion); micro P/R/F1 (+CIs) and exact match each.
    Counters: n_total, n_scored, n_gt_empty_model_present, n_both_empty, n_parse_error, n_truncated.
    """
    cfg_raw = {}
    if config:
        cfg_raw = (config.get("task_settings", {}) or {}).get("label_extraction_mamma", {}) or {}

    acr_range        = cfg_raw.get("acr_range",            "min")
    birads6_handling = cfg_raw.get("birads6_handling",      "map_to_5")
    gt_empty_mode    = cfg_raw.get("gt_empty_ext_present",  "ignore")
    gt_empty_fp      = gt_empty_mode == "fp"

    norms = _mamma_norms()
    df = pd.read_csv(results_csv_path, dtype=str).fillna("")

    n_total      = len(df)
    n_parse_error = int((df["parse_error"].str.lower() == "true").sum())
    n_truncated, n_parse_error_truncated = _truncation_counts(df)

    field_data, lesion_sides = _score_mamma(df, norms, acr_range, birads6_handling, gt_empty_fp)

    # Sensitivity variants
    b6_alt, b6_tag, b6_variant = _BIRADS6_VARIANTS.get(birads6_handling, _BIRADS6_VARIANTS["map_to_5"])
    ge_alt, ge_tag, ge_variant = _GT_EMPTY_VARIANTS.get(gt_empty_mode, _GT_EMPTY_VARIANTS["ignore"])
    b6_fields, _ = _score_mamma(df, norms, acr_range, b6_alt, gt_empty_fp)
    ge_fields, ge_lesions = _score_mamma(df, norms, acr_range, birads6_handling, ge_alt)

    # Model answers "BI-RADS 6" (the GT has no 6)
    n_birads6 = {}
    for side in ("birads_li", "birads_re"):
        all_6 = scored_6 = 0
        for _, row in df.iterrows():
            if row["parse_error"].lower() == "true":
                continue
            if _normalize_birads(row.get(f"model_{side}", ""), norms["birads"], "keep") == "6":
                all_6 += 1
                scored_6 += _normalize_birads(row.get(f"gt_{side}", ""), norms["birads"], "keep") is not None
        n_birads6[side] = (all_6, scored_6)

    acr_exam = _acr_exam_level(df, norms, acr_range)
    meno_rep = _menopause_in_report_level(df, norms, acr_range)

    # ─── Metrics ──────────────────────────────────────────────────────────────
    result: dict = {"path": out_path}
    summaries = {f: _mamma_field_summary(field_data[f]) for f in _MAMMA_FIELDS}
    for field, s in summaries.items():
        if s["accuracy"] is not None:
            result[f"{field}_accuracy_pct"] = s["accuracy"]
            result[f"{field}_macro_f1_pct"] = s["macro_f1"]
    lesion_summaries = {k: _mamma_lesion_summary(sd) for k, sd in lesion_sides.items()}
    for (side_key, mode), ls in lesion_summaries.items():
        # Main metric (type) keeps the plain key names; count view is prefixed.
        prefix = side_key if mode == "type" else f"{side_key}_count"
        if ls["defined"]:
            result[f"{prefix}_micro_f1_pct"] = round(ls["f1_raw"] * 100, 2)
        result[f"{prefix}_exact_match_pct"] = ls["exact"]
    for field, s in summaries.items():
        if s["coverage"] is not None:
            result[f"{field}_coverage_pct"] = s["coverage"]
        if s["accuracy_answered"] is not None:
            result[f"{field}_accuracy_when_answered_pct"] = s["accuracy_answered"]
    if acr_exam["accuracy"] is not None:
        result["acr_exam_accuracy_pct"] = acr_exam["accuracy"]
    if meno_rep["accuracy"] is not None:
        result["menopause_accuracy_in_report_pct"] = meno_rep["accuracy"]
    b6_summaries = {f: _mamma_field_summary(b6_fields[f]) for f in ("birads_li", "birads_re")}
    for field, s in b6_summaries.items():
        if s["accuracy"] is not None:
            result[f"{field}_accuracy_{b6_tag}_pct"] = s["accuracy"]
            result[f"{field}_macro_f1_{b6_tag}_pct"] = s["macro_f1"]
        result[f"{field}_n_model_birads6"] = n_birads6[field][0]
    ge_summaries = {f: _mamma_field_summary(ge_fields[f]) for f in _MAMMA_FIELDS}
    for field, s in ge_summaries.items():
        if s["accuracy"] is not None:
            result[f"{field}_accuracy_{ge_tag}_pct"] = s["accuracy"]
            result[f"{field}_macro_f1_{ge_tag}_pct"] = s["macro_f1"]
    ge_lesion_summaries = {k: _mamma_lesion_summary(sd) for k, sd in ge_lesions.items()}
    for (side_key, mode), ls in ge_lesion_summaries.items():
        if mode == "type" and ls["defined"]:
            result[f"{side_key}_micro_f1_{ge_tag}_pct"] = ls["f1"]
    if not gt_empty_fp:
        for field in _MAMMA_FIELDS:
            result[f"{field}_n_ignored_model_values"] = field_data[field]["n_gt_empty_model_present"]
    if n_truncated is not None:
        result["n_truncated"] = n_truncated

    with open(out_path, "w", encoding="utf-8") as f:
        def _w(obj):
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        _w({"type": "metric", "metric": "n_total",     "value": n_total})
        _w({"type": "metric", "metric": "n_parse_error","value": n_parse_error})
        _w({"type": "metric", "metric": "n_truncated", "value": n_truncated,
            "n_parse_error_truncated": n_parse_error_truncated,
            "note": ("finish_reason == 'length' (answer cut off at max_tokens)"
                     if n_truncated is not None else "finish_reason not recorded")})

        for field in _MAMMA_FIELDS:
            fd, s = field_data[field], summaries[field]
            _w({"type": "metric", "field": field, "metric": "n_scored",
                "value": fd["n_scored"]})
            _w({"type": "metric", "field": field, "metric": "n_gt_empty_model_present",
                "value": fd["n_gt_empty_model_present"],
                "note": ("GT empty, model gave a value: "
                         + ("scored as wrong (gt_empty_ext_present=fp)" if gt_empty_fp
                            else "not scored (gt_empty_ext_present=ignore)"))})
            _w({"type": "metric", "field": field, "metric": "n_both_empty",
                "value": fd["n_both_empty"]})
            acc_row = {"type": "metric", "field": field, "metric": "accuracy_pct",
                       "value": s["accuracy"], "ci_lo": s["acc_lo"], "ci_hi": s["acc_hi"]}
            if field.startswith("acr"):
                acc_row["note"] = _ACR_SIDE_NOTE
            _w(acc_row)
            _w({"type": "metric", "field": field, "metric": "macro_f1_pct", "value": s["macro_f1"],
                "note": _MACRO_F1_NOTE})
            _w({"type": "metric", "field": field, "metric": "coverage_pct", "value": s["coverage"],
                "n_model_missing": fd["n_model_missing"],
                "note": "share of scored items where the model gave a value"})
            _w({"type": "metric", "field": field, "metric": "accuracy_when_answered_pct",
                "value": s["accuracy_answered"], "n": s["n_answered"],
                "note": "accuracy among scored items where the model gave a value"})

            for gt_cls, preds in s["confusion"].items():
                for pred_cls, count in preds.items():
                    if count:
                        _w({"type": "confusion", "field": field,
                            "gt": gt_cls, "model": pred_cls, "count": count})

        _w({"type": "metric", "field": "acr_exam", "metric": "accuracy_pct",
            "value": acr_exam["accuracy"], "ci_lo": acr_exam["ci_lo"], "ci_hi": acr_exam["ci_hi"],
            "n_exams": acr_exam["n_exams"],
            "n_exams_gt_li_ne_re": acr_exam["n_gt_differs"],
            "n_exams_gt_one_side": acr_exam["n_one_side"],
            "n_exams_model_li_ne_re": acr_exam["n_model_differs"],
            "note": ("one BPE decision per exam; GT value = li if li == re (or the only GT side), "
                     "exams with GT li != re excluded; correct iff all non-empty model sides "
                     "equal the GT value")})
        if meno_rep["accuracy"] is not None:
            _w({"type": "metric", "field": "menopause", "metric": "accuracy_in_report_pct",
                "value": meno_rep["accuracy"], "ci_lo": meno_rep["ci_lo"], "ci_hi": meno_rep["ci_hi"],
                "n": meno_rep["n"], "n_not_in_report": meno_rep["n_not_in_report"],
                "note": ("menopause accuracy on exams whose report text mentions the status "
                         "(keyword match); the annotation often takes the status from other "
                         "sources, so menopause_accuracy_pct also counts 'not in the report'")})

        for (side_key, mode), sd in lesion_sides.items():
            ls = lesion_summaries[(side_key, mode)]
            base = {"type": "metric", "field": side_key, "lesion_view": mode,
                    "n_sides": sd["n"], "n_ignored_gt_empty": sd["ignored"]}
            _w({**base, "metric": "micro_precision_pct", "value": ls["precision"],
                "ci_lo": ls["precision_ci"][0], "ci_hi": ls["precision_ci"][1]})
            _w({**base, "metric": "micro_recall_pct", "value": ls["recall"],
                "ci_lo": ls["recall_ci"][0], "ci_hi": ls["recall_ci"][1]})
            _w({**base, "metric": "micro_f1_pct", "value": ls["f1"],
                "ci_lo": ls["f1_ci"][0], "ci_hi": ls["f1_ci"][1],
                "note": "CI: bootstrap over reports, micro-F1 recomputed per resample"})
            _w({**base, "metric": "exact_match_pct", "value": ls["exact"]})
            _w({**base, "metric": "n_ignored_sides_model_present", "value": sd["ignored_model_present"],
                "n_ignored_model_entries": sd["ignored_model_entries"],
                "note": "GT side without lesions, model listed lesions (not scored)"
                        if not gt_empty_fp else "always 0 with gt_empty_ext_present=fp"})

        # ─── Sensitivity analyses ───
        for field in ("birads_li", "birads_re"):
            s = b6_summaries[field]
            base = {"type": "metric", "field": field, "variant": b6_variant,
                    "n_scored": b6_fields[field]["n_scored"],
                    "note": f"sensitivity analysis: birads6_handling={b6_alt} "
                            f"(primary: {birads6_handling})"}
            _w({**base, "metric": f"accuracy_{b6_tag}_pct", "value": s["accuracy"],
                "ci_lo": s["acc_lo"], "ci_hi": s["acc_hi"]})
            _w({**base, "metric": f"macro_f1_{b6_tag}_pct", "value": s["macro_f1"]})
            _w({"type": "metric", "field": field, "metric": "n_model_birads6",
                "value": n_birads6[field][0], "n_model_birads6_gt_present": n_birads6[field][1],
                "note": "model answered BI-RADS 6 (GT has no 6); "
                        f"primary birads6_handling={birads6_handling}"})

        for field in _MAMMA_FIELDS:
            s = ge_summaries[field]
            base = {"type": "metric", "field": field, "variant": ge_variant,
                    "n_scored": ge_fields[field]["n_scored"],
                    "note": f"sensitivity analysis: gt_empty_ext_present="
                            f"{'fp' if ge_alt else 'ignore'} (primary: {gt_empty_mode})"
                            + ("; worst-case bound: empty GT fields are mostly 'not annotated', so "
                               "this also penalises values that are correct in the report"
                               if ge_alt else "")}
            _w({**base, "metric": f"accuracy_{ge_tag}_pct", "value": s["accuracy"],
                "ci_lo": s["acc_lo"], "ci_hi": s["acc_hi"]})
            _w({**base, "metric": f"macro_f1_{ge_tag}_pct", "value": s["macro_f1"]})
        for (side_key, mode), sd in ge_lesions.items():
            ls = ge_lesion_summaries[(side_key, mode)]
            base = {"type": "metric", "field": side_key, "lesion_view": mode,
                    "variant": ge_variant, "n_sides": sd["n"],
                    "note": f"sensitivity analysis: gt_empty_ext_present="
                            f"{'fp' if ge_alt else 'ignore'} (primary: {gt_empty_mode})"
                            + ("; worst-case bound: empty GT fields are mostly 'not annotated', so "
                               "this also penalises values that are correct in the report"
                               if ge_alt else "")}
            _w({**base, "metric": f"micro_precision_{ge_tag}_pct", "value": ls["precision"],
                "ci_lo": ls["precision_ci"][0], "ci_hi": ls["precision_ci"][1]})
            _w({**base, "metric": f"micro_recall_{ge_tag}_pct", "value": ls["recall"],
                "ci_lo": ls["recall_ci"][0], "ci_hi": ls["recall_ci"][1]})
            _w({**base, "metric": f"micro_f1_{ge_tag}_pct", "value": ls["f1"],
                "ci_lo": ls["f1_ci"][0], "ci_hi": ls["f1_ci"][1]})
            _w({**base, "metric": f"exact_match_{ge_tag}_pct", "value": ls["exact"]})

        for _, row in df.iterrows():
            obj = {"type": "item"}
            for col, val in row.to_dict().items():
                obj[col] = _jsonable(val)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if logger:
        logger.verbose("\n--- Mamma-MRT Extraction Evaluation ---")
        logger.verbose(f"n_total={n_total}  n_parse_error={n_parse_error}  n_truncated={n_truncated}")
        for field in _MAMMA_FIELDS:
            s = summaries[field]
            nb  = field_data[field]["n_scored"]
            logger.verbose(f"  {field:<12}  acc={s['accuracy']}%  macro_f1={s['macro_f1']}%  "
                           f"coverage={s['coverage']}%  n={nb}")
        logger.verbose(f"  acr_exam      acc={acr_exam['accuracy']}%  n={acr_exam['n_exams']}  "
                       f"(GT li!=re excluded: {acr_exam['n_gt_differs']})")
        for side_key in ("lesions_li", "lesions_re"):
            logger.verbose(
                f"  {side_key:<12}  types: micro_f1={result.get(f'{side_key}_micro_f1_pct')}%  "
                f"exact={result.get(f'{side_key}_exact_match_pct')}%  |  "
                f"count: micro_f1={result.get(f'{side_key}_count_micro_f1_pct')}%  "
                f"exact={result.get(f'{side_key}_count_exact_match_pct')}%"
            )
        logger.verbose(f"  sensitivity {b6_variant}: " + "  ".join(
            f"{fl} acc={b6_summaries[fl]['accuracy']}%" for fl in ("birads_li", "birads_re")))
        logger.verbose(f"  sensitivity {ge_variant}: " + "  ".join(
            f"{fl} acc={ge_summaries[fl]['accuracy']}%" for fl in _MAMMA_FIELDS))

    return result


def print_mamma_extraction_terminal_report(
    results_csv_path: str, report: dict = None
) -> None:
    r = report or {}
    for field in _MAMMA_FIELDS:
        acc = r.get(f"{field}_accuracy_pct", "?")
        mf1 = r.get(f"{field}_macro_f1_pct", "?")
        cov = r.get(f"{field}_coverage_pct", "?")
        print(f"  {field:<12}  acc={acc}%  macro_f1={mf1}%  coverage={cov}%")
    print(f"  {'acr_exam':<12}  acc={r.get('acr_exam_accuracy_pct', '?')}%")
    if r.get("menopause_accuracy_in_report_pct") is not None:
        print(f"  {'menopause':<12}  acc={r['menopause_accuracy_in_report_pct']}% "
              "where the report states the status")
    for side in ("lesions_li", "lesions_re"):
        print(
            f"  {side:<12}  types: micro_f1={r.get(f'{side}_micro_f1_pct', '?')}%  "
            f"exact={r.get(f'{side}_exact_match_pct', '?')}%  |  "
            f"count: micro_f1={r.get(f'{side}_count_micro_f1_pct', '?')}%  "
            f"exact={r.get(f'{side}_count_exact_match_pct', '?')}%"
        )
    sens = {k: v for k, v in r.items()
            if any(t in k for t in ("_birads6keep_", "_birads6mapto5_", "_gtemptyfp_", "_gtemptyignore_"))
            and "_accuracy_" in k}
    if sens:
        print("  sensitivity: " + "  ".join(f"{k}={v}%" for k, v in sens.items()))


# ===========================================================================
# Arm X-ray Label Extraction Evaluation
# ===========================================================================

_ARM_ACCURACY_NOTE = (
    "secondary metric: mean per-report share of correct label decisions; dominated by "
    "true negatives (most labels are absent) – compare all_negative_baseline_accuracy_pct"
)
_VERBATIM_NOTE = (
    "share of citations (labels marked present, citation given) that occur verbatim in the "
    "report; verbatim != evidence, see verbatim_rate_true_positive_pct / _false_positive_pct"
)


def _binary_prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def _mcc(tp, fp, fn, tn) -> float:
    """Matthews correlation coefficient; 0.0 if undefined (a marginal is zero)."""
    tp, fp, fn, tn = (float(x) for x in (tp, fp, fn, tn))
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


def _arm_aggregate(label_stats: dict) -> dict:
    """Micro-/Macro-F1 and pooled MCC over a set of per-label TP/FP/FN/TN counts.

    Labels with tp+fp+fn == 0 (no positives in GT, none predicted) have an
    undefined F1 and are excluded from the macro average.
    """
    tp = sum(s["tp"] for s in label_stats.values())
    fp = sum(s["fp"] for s in label_stats.values())
    fn = sum(s["fn"] for s in label_stats.values())
    tn = sum(s["tn"] for s in label_stats.values())
    _, _, micro_f1 = _binary_prf(tp, fp, fn)
    defined = [_binary_prf(s["tp"], s["fp"], s["fn"])[2]
               for s in label_stats.values() if s["tp"] + s["fp"] + s["fn"] > 0]
    macro_f1 = sum(defined) / len(defined) if defined else 0.0
    return {"micro_f1": micro_f1, "macro_f1": macro_f1, "mcc": _mcc(tp, fp, fn, tn),
            "n_labels": len(label_stats), "n_labels_defined": len(defined)}


def _arm_bootstrap(reports: list, keys: list, n_boot: int = 1000, alpha: float = 0.05) -> dict:
    """
    95% CIs for micro-F1, macro-F1 and MCC: resample reports (seed 42, same draws as
    _bootstrap_micro_prf_ci), recompute from the summed per-label counts.
    reports: list of {key: (tp, fp, fn, tn)}. Returns {"micro_f1"|"macro_f1"|"mcc": (lo, hi)}.
    """
    import numpy as _np
    out = {"micro_f1": (0.0, 0.0), "macro_f1": (0.0, 0.0), "mcc": (0.0, 0.0)}
    n = len(reports)
    if n == 0 or not keys:
        return out
    col = {k: i for i, k in enumerate(keys)}
    mats = _np.zeros((4, n, len(keys)), dtype=_np.int64)
    for r_i, rep in enumerate(reports):
        for key, counts in rep.items():
            for c_i in range(4):
                mats[c_i, r_i, col[key]] = counts[c_i]
    rng = random.Random(42)
    micro, macro, mccs = [], [], []
    for _ in range(n_boot):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        tp, fp, fn, tn = (m[idx].sum(axis=0) for m in mats)
        T, F, N, TN = int(tp.sum()), int(fp.sum()), int(fn.sum()), int(tn.sum())
        micro.append(2 * T / (2 * T + F + N) if (2 * T + F + N) > 0 else 0.0)
        denom = 2 * tp + fp + fn
        defined = denom > 0
        macro.append(float((2 * tp[defined] / denom[defined]).mean()) if defined.any() else 0.0)
        mccs.append(_mcc(T, F, N, TN))
    lo_i, hi_i = int(n_boot * alpha / 2), int(n_boot * (1 - alpha / 2)) - 1
    for name, vals in (("micro_f1", micro), ("macro_f1", macro), ("mcc", mccs)):
        vals.sort()
        out[name] = (vals[lo_i], vals[hi_i])
    return out


def write_arm_extraction_report_jsonl(
    results_csv_path: str,
    out_path: str,
    logger=None,
    config: dict = None,
) -> dict:
    """
    Evaluate an Arm X-ray extraction results CSV.

    Labels are kept per region ("<region> | <label>"), because labels with the same
    name (e.g. "Foreign Bodies") are different tasks in different regions.

    Metrics:
    - F1, sensitivity, specificity, MCC per label
    - micro-/macro-F1 and MCC overall and per region (macro only over labels with a
      defined F1), each with a 95% bootstrap CI (1000 resamples over reports, seed 42)
    - accuracy (secondary; share of correct label decisions per report, averaged) and
      all_negative_baseline_accuracy_pct (same averaging, prediction "all negative")
    - verbatim_citation_rate_pct: share of citations (finding=true) that occur verbatim
      in the report (checked in the task, column citation_check_json), split into
      true-positive and false-positive calls
    - No extraction = "nothing found": an unparseable answer (n_parse_error) and labels
      missing from an answer (n_missing_labels, stored as finding=null) count as negative
      predictions (FN for a positive GT label, TN otherwise) – as for Mamma, where a
      missing value counts as wrong. Both counts are reported.
    """
    df = pd.read_csv(results_csv_path, dtype=str).fillna("")

    n_total      = len(df)
    n_parse_error = int((df["parse_error"].str.lower() == "true").sum())
    n_truncated, n_parse_error_truncated = _truncation_counts(df)
    has_citation_col = "citation_check_json" in df.columns

    label_stats: dict = {}
    region_of: dict = {}
    reports: list = []          # per scored report: {key: (tp, fp, fn, tn)}
    report_region: list = []
    per_report_acc: list = []
    per_report_allneg: list = []
    n_missing = n_missing_pos = n_reports_missing = 0
    cite = {"all": [0, 0], "tp": [0, 0], "fp": [0, 0]}   # [verbatim, total]
    n_present_calls = 0

    for _, row in df.iterrows():
        # no extraction (unparseable answer) = nothing found: scored as all-negative
        is_parse_error = row["parse_error"].lower() == "true"
        region = row.get("region", "") or "unknown"

        try:
            gt_labels = json.loads(row.get("gt_labels_json") or "{}")
        except Exception:
            gt_labels = {}
        try:
            model_labels = json.loads(row.get("model_labels_json") or "{}")
        except Exception:
            model_labels = {}
        if not isinstance(model_labels, dict) or is_parse_error:
            model_labels = {}

        rep: dict = {}
        row_correct = row_neg = row_scored = row_missing = 0
        gt_bins: dict = {}
        for label, gt_val in gt_labels.items():
            try:
                gt_bin = int(gt_val)
            except (ValueError, TypeError):
                gt_bin = 0
            gt_bins[label] = gt_bin
            entry = model_labels.get(label)
            finding = entry.get("finding") if isinstance(entry, dict) else None
            if finding is None:
                if not is_parse_error:       # a label left out of a parsed answer
                    row_missing += 1
                    n_missing_pos += gt_bin
                finding = False              # no extraction = not found
            pred_bin = int(finding is True)
            n_present_calls += pred_bin

            key = f"{region} | {label}"
            region_of[key] = region
            s = label_stats.setdefault(key, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            outcome = ("tp" if gt_bin and pred_bin else "fp" if pred_bin
                       else "fn" if gt_bin else "tn")
            s[outcome] += 1
            rep[key] = (int(outcome == "tp"), int(outcome == "fp"),
                        int(outcome == "fn"), int(outcome == "tn"))
            row_correct += int(gt_bin == pred_bin)
            row_neg += int(gt_bin == 0)
            row_scored += 1

        n_missing += row_missing
        n_reports_missing += int(row_missing > 0)
        reports.append(rep)
        report_region.append(region)
        if row_scored:
            per_report_acc.append(row_correct / row_scored)
            per_report_allneg.append(row_neg / row_scored)

        if has_citation_col:
            try:
                checks = json.loads(row.get("citation_check_json") or "{}")
            except Exception:
                checks = {}
            for label, ok in (checks or {}).items():
                kind = "tp" if gt_bins.get(label, 0) else "fp"
                for k in ("all", kind):
                    cite[k][0] += int(bool(ok))
                    cite[k][1] += 1

    keys = sorted(label_stats)
    overall = _arm_aggregate(label_stats)
    boot = _arm_bootstrap(reports, keys)
    acc_mean = sum(per_report_acc) / len(per_report_acc) if per_report_acc else 0.0
    allneg_mean = sum(per_report_allneg) / len(per_report_allneg) if per_report_allneg else 0.0

    def _rate(k):
        return round(cite[k][0] / cite[k][1] * 100, 2) if cite[k][1] else None

    cite_pct = _rate("all") if has_citation_col else None

    def _p(x):
        return round(x * 100, 2)

    result: dict = {
        "path":               out_path,
        "n_total":           n_total,
        "n_parse_error":      n_parse_error,
        "micro_f1_pct":       _p(overall["micro_f1"]),
        "macro_f1_pct":       _p(overall["macro_f1"]),
        "mcc":                round(overall["mcc"], 4),
        "accuracy_pct":       _p(acc_mean),
        "all_negative_baseline_accuracy_pct": _p(allneg_mean),
        "n_missing_labels":   n_missing,
    }
    if n_truncated is not None:
        result["n_truncated"] = n_truncated
    if cite_pct is not None:
        result["verbatim_citation_rate_pct"] = cite_pct
        for kind, name in (("tp", "verbatim_rate_true_positive_pct"),
                           ("fp", "verbatim_rate_false_positive_pct")):
            if _rate(kind) is not None:
                result[name] = _rate(kind)

    regions = sorted(set(region_of.values()))
    region_results = {}
    for region in regions:
        agg = _arm_aggregate({k: v for k, v in label_stats.items() if region_of[k] == region})
        r_reports = [rep for rep, reg in zip(reports, report_region) if reg == region]
        r_keys = [k for k in keys if region_of[k] == region]
        agg["ci"] = _arm_bootstrap(r_reports, r_keys)
        agg["n_reports"] = len(r_reports)
        region_results[region] = agg
        result[f"{region}_micro_f1_pct"] = _p(agg["micro_f1"])
        result[f"{region}_macro_f1_pct"] = _p(agg["macro_f1"])
        result[f"{region}_mcc"] = round(agg["mcc"], 4)

    with open(out_path, "w", encoding="utf-8") as f:
        def _w(obj):
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        ci_note = "CI: bootstrap over reports (1000 resamples, seed 42), recomputed per resample"
        _w({"type": "metric", "metric": "n_total",      "value": n_total})
        _w({"type": "metric", "metric": "n_parse_error", "value": n_parse_error,
            "note": "reports whose answer could not be parsed; scored as all-negative "
                    "(no extraction = nothing found)"})
        _w({"type": "metric", "metric": "n_truncated", "value": n_truncated,
            "n_parse_error_truncated": n_parse_error_truncated,
            "note": ("finish_reason == 'length' (answer cut off at max_tokens)"
                     if n_truncated is not None else "finish_reason not recorded")})
        _w({"type": "metric", "metric": "n_missing_labels", "value": n_missing,
            "n_missing_labels_gt_positive": n_missing_pos,
            "n_reports_with_missing_labels": n_reports_missing,
            "note": "labels absent from a parsed answer; scored as negative "
                    "(no extraction = nothing found)"})
        _w({"type": "metric", "metric": "micro_f1_pct",  "value": result["micro_f1_pct"],
            "ci_lo": _p(boot["micro_f1"][0]), "ci_hi": _p(boot["micro_f1"][1]),
            "note": ci_note})
        _w({"type": "metric", "metric": "macro_f1_pct",  "value": result["macro_f1_pct"],
            "ci_lo": _p(boot["macro_f1"][0]), "ci_hi": _p(boot["macro_f1"][1]),
            "n_labels": overall["n_labels"], "n_labels_defined": overall["n_labels_defined"],
            "note": ci_note + "; labels without positives in GT and prediction excluded"})
        _w({"type": "metric", "metric": "mcc", "value": result["mcc"],
            "ci_lo": round(boot["mcc"][0], 4), "ci_hi": round(boot["mcc"][1], 4),
            "note": "Matthews correlation over all pooled label decisions (range -1..1); " + ci_note})
        _w({"type": "metric", "metric": "accuracy_pct",  "value": result["accuracy_pct"],
            "secondary": True, "note": _ARM_ACCURACY_NOTE})
        _w({"type": "metric", "metric": "all_negative_baseline_accuracy_pct",
            "value": result["all_negative_baseline_accuracy_pct"],
            "note": "accuracy_pct of a model that marks every label absent"})
        _w({"type": "metric", "metric": "verbatim_citation_rate_pct", "value": cite_pct,
            "n_citations": cite["all"][1], "n_present_calls": n_present_calls,
            "note": _VERBATIM_NOTE})
        _w({"type": "metric", "metric": "verbatim_rate_true_positive_pct",
            "value": _rate("tp") if has_citation_col else None, "n_citations": cite["tp"][1],
            "note": "verbatim share among citations of true-positive present-calls"})
        _w({"type": "metric", "metric": "verbatim_rate_false_positive_pct",
            "value": _rate("fp") if has_citation_col else None, "n_citations": cite["fp"][1],
            "note": "verbatim share among citations of false-positive present-calls"})

        for region, agg in region_results.items():
            ci = agg["ci"]
            _w({"type": "region_metric", "region": region,
                "micro_f1_pct": _p(agg["micro_f1"]),
                "macro_f1_pct": _p(agg["macro_f1"]),
                "mcc": round(agg["mcc"], 4),
                "n_labels": agg["n_labels"], "n_labels_defined": agg["n_labels_defined"],
                "n_reports": agg["n_reports"]})
            for metric, key in (("micro_f1_pct", "micro_f1"), ("macro_f1_pct", "macro_f1")):
                _w({"type": "metric", "region": region, "metric": metric,
                    "value": _p(agg[key]), "ci_lo": _p(ci[key][0]), "ci_hi": _p(ci[key][1]),
                    "n_reports": agg["n_reports"], "note": ci_note})
            _w({"type": "metric", "region": region, "metric": "mcc",
                "value": round(agg["mcc"], 4), "ci_lo": round(ci["mcc"][0], 4),
                "ci_hi": round(ci["mcc"][1], 4), "n_reports": agg["n_reports"], "note": ci_note})

        for key in keys:
            s = label_stats[key]
            tp, fp, fn, tn = s["tp"], s["fp"], s["fn"], s["tn"]
            _, sens, f1 = _binary_prf(tp, fp, fn)
            defined = tp + fp + fn > 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else None
            _w({"type": "label_metric", "region": region_of[key], "label": key.split(" | ", 1)[1],
                "f1_pct": round(f1 * 100, 2) if defined else None,
                "sensitivity_pct": round(sens * 100, 2) if (tp + fn) > 0 else None,
                "specificity_pct": round(spec * 100, 2) if spec is not None else None,
                "mcc": round(_mcc(tp, fp, fn, tn), 4),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn})

        for _, row in df.iterrows():
            obj = {"type": "item"}
            for col, val in row.to_dict().items():
                obj[col] = _jsonable(val)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if logger:
        logger.verbose("\n--- Arm X-ray Extraction Evaluation ---")
        logger.verbose(
            f"n={n_total}  parse_err={n_parse_error}  truncated={n_truncated}  "
            f"missing_labels={n_missing}  "
            f"micro_f1={result['micro_f1_pct']:.1f}%  macro_f1={result['macro_f1_pct']:.1f}%  "
            f"mcc={result['mcc']:.3f}  acc={result['accuracy_pct']:.1f}% "
            f"(all-negative {result['all_negative_baseline_accuracy_pct']:.1f}%)  "
            f"verbatim={cite_pct}"
        )
        for region, agg in region_results.items():
            logger.verbose(f"  {region:<9} micro_f1={agg['micro_f1']*100:.1f}%  "
                           f"macro_f1={agg['macro_f1']*100:.1f}%  mcc={agg['mcc']:.3f}")

    return result


def print_arm_extraction_terminal_report(
    results_csv_path: str, report: dict = None
) -> None:
    r = report or {}
    cite = r.get("verbatim_citation_rate_pct")
    print(
        f"  Micro-F1: {r.get('micro_f1_pct', '?')}%  "
        f"Macro-F1: {r.get('macro_f1_pct', '?')}%  "
        f"MCC: {r.get('mcc', '?')}  "
        f"Accuracy (secondary): {r.get('accuracy_pct', '?')}% "
        f"[all-negative: {r.get('all_negative_baseline_accuracy_pct', '?')}%]  "
        f"Verbatim citations: {f'{cite}%' if cite is not None else 'n/a'}  "
        f"(n={r.get('n_total', '?')}, parse_err={r.get('n_parse_error', '?')}, "
        f"missing_labels={r.get('n_missing_labels', '?')}, truncated={r.get('n_truncated', 'n/a')})"
    )
    for region in ("clavicle", "elbow", "thumb"):
        if f"{region}_micro_f1_pct" in r:
            print(f"    {region:<9} Micro-F1: {r[f'{region}_micro_f1_pct']}%  "
                  f"Macro-F1: {r[f'{region}_macro_f1_pct']}%  MCC: {r.get(f'{region}_mcc')}")


# ===========================================================================
# Command line
# ===========================================================================

def _load_eval_config(path: str = None) -> dict:
    """Config for CLI re-evaluation: --config path, else config.yaml, else config.default.yaml."""
    import yaml
    if path is None:
        path = next((p for p in ("config.yaml", "config.default.yaml") if os.path.exists(p)), None)
    if path is None:
        print("No config.yaml/config.default.yaml found – using default task_settings.")
        return {}
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    print(f"task_settings from: {path}")
    return cfg


def _load_judge_client_from_config(judge_model: Optional[str] = None, config_path: Optional[str] = None):
    """
    Judge client from the config's judge: section, built exactly like in main.py
    (judge.temperature / max_tokens / seed / extra_body). *config_path* is the
    --config file (default config.yaml, else config.default.yaml); *judge_model*
    overrides judge.model_name, e.g. to run a second judge for an agreement
    analysis (verdicts go to a separate cache file per judge model).
    """
    from main import _build_judge_client
    cfg = _load_eval_config(config_path)
    if not cfg.get("judge"):
        raise SystemExit("No 'judge:' section in the config – cannot run LLM-as-a-Judge.")
    if judge_model:
        cfg = dict(cfg)
        cfg["judge"] = dict(cfg["judge"], model_name=judge_model)
    client = _build_judge_client(cfg)
    if client is None:
        raise SystemExit("Could not create the judge client (see warning above).")
    return client, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a benchmark results CSV.")
    parser.add_argument("csv", help="Results CSV, e.g. results/<run>/medqa_results.csv")
    parser.add_argument(
        "--out",
        default="results/benchmark_report.jsonl",
        help="Output JSONL report path (default: results/benchmark_report.jsonl)",
    )
    parser.add_argument(
        "--type",
        dest="eval_type",
        choices=["mcq", "vqa", "mamma_extraction", "arm_extraction"],
        default="mcq",
        help="Evaluation type",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config for task_settings (mamma_extraction/arm_extraction); "
             "default: config.yaml, else config.default.yaml.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        default=False,
        help="Run LLM-as-a-Judge for open-ended answers (judge: section of config.yaml + live LLM).",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Override judge.model_name (implies --judge); verdicts are cached per judge model.",
    )
    args = parser.parse_args()
    run_judge = args.judge or bool(args.judge_model)

    if args.eval_type == "mcq":
        report = write_report_jsonl(args.csv, out_path=args.out)
        print_terminal_report(args.csv)
        print(f"Wrote: {report['path']}")

    elif args.eval_type == "vqa":
        client, cfg = _load_judge_client_from_config(args.judge_model, args.config) if run_judge else (None, None)
        report = write_vqa_report_jsonl(args.csv, out_path=args.out, client=client,
                                        run_judge=run_judge, config=cfg)
        print_vqa_terminal_report(args.csv, report=report)
        print(f"Wrote: {report['path']}")

    elif args.eval_type == "mamma_extraction":
        report = write_mamma_extraction_report_jsonl(
            args.csv, out_path=args.out, config=_load_eval_config(args.config))
        print_mamma_extraction_terminal_report(args.csv, report=report)
        print(f"Wrote: {report['path']}")

    elif args.eval_type == "arm_extraction":
        report = write_arm_extraction_report_jsonl(
            args.csv, out_path=args.out, config=_load_eval_config(args.config))
        print_arm_extraction_terminal_report(args.csv, report=report)
        print(f"Wrote: {report['path']}")


if __name__ == "__main__":
    main()
