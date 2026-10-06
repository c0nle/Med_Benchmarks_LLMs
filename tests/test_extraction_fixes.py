"""
Tests for the review fixes of the Mamma-MRT / Arm X-ray label-extraction benchmarks
(concurrent runner, resume, last_meta columns, sensitivity analyses, Arm metrics, CLI).

Synthetic data only – no patient reports.
"""
import json
import os
import sys
import threading
import time

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.client import ServerUnavailableError  # noqa: E402
from evaluate import (  # noqa: E402
    _build_normalizer,
    _load_mamma_norm,
    write_arm_extraction_report_jsonl,
    write_mamma_extraction_report_jsonl,
)
from tasks import _extraction_runner as runner  # noqa: E402
from tasks import arm_extraction, mamma_extraction  # noqa: E402


# ---------------------------------------------------------------------------
# Fake client (thread-local last_meta like core.client)
# ---------------------------------------------------------------------------

class FakeClient:
    def __init__(self, answer_fn):
        self.answer_fn = answer_fn
        self._local = threading.local()
        self._lock = threading.Lock()
        self.system_prompts = []

    @property
    def last_meta(self):
        return getattr(self._local, "meta", None)

    def ask_question(self, prompt, system_prompt=None):
        with self._lock:
            self.system_prompts.append(system_prompt)
        answer, meta = self.answer_fn(prompt)
        time.sleep(0.002)  # let threads interleave
        self._local.meta = meta
        return answer


def _case(prompt):
    """Synthetic case number embedded in the report text as 'CASE-<n>.'"""
    return int(prompt.split("CASE-")[1].split(".")[0])


def _mamma_items(n):
    return [{"id": f"m{i}", "benchmark": "LabelExtractionMamma",
             "text": f"Synthetischer Befund CASE-{i}. Keine echten Daten.",
             "gt": {"menopause": "post", "birads_li": "2", "birads_re": None,
                    "acr_li": "1", "acr_re": "1", "lesions_li": ["Zyste"], "lesions_re": []}}
            for i in range(n)]


def _mamma_answer(prompt):
    i = _case(prompt)
    ans = {"menopause": "post", "links": {"birads": 2, "acr": 1, "lesionen": ["Zyste"]},
           "rechts": {"birads": None, "acr": 1, "lesionen": []}}
    meta = {"finish_reason": "length" if i % 3 == 0 else "stop", "completion_tokens": 100 + i}
    return json.dumps(ans), meta


def _read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# ---------------------------------------------------------------------------
# Runner: concurrency, meta columns, resume, errors
# ---------------------------------------------------------------------------

def test_concurrent_run_writes_each_item_once_with_own_meta(tmp_path):
    path = str(tmp_path / "m.csv")
    client = FakeClient(_mamma_answer)
    cfg = {"benchmark_settings": {"concurrency": 4}}
    mamma_extraction.run(cfg, client, _mamma_items(30), path)
    df = _read(path)
    assert sorted(df["id"]) == sorted(f"m{i}" for i in range(30))
    for _, row in df.iterrows():
        i = int(row["id"][1:])
        assert row["completion_tokens"] == str(100 + i)
        assert row["finish_reason"] == ("length" if i % 3 == 0 else "stop")
    assert set(client.system_prompts) == {mamma_extraction._SYSTEM_PROMPT_DE}


def test_meta_columns_empty_without_last_meta(tmp_path):
    class NoMeta:
        def ask_question(self, prompt, system_prompt=None):
            return _mamma_answer(prompt)[0]

    path = str(tmp_path / "m.csv")
    mamma_extraction.run({}, NoMeta(), _mamma_items(2), path)
    df = _read(path)
    assert list(df["finish_reason"]) == ["", ""] and list(df["completion_tokens"]) == ["", ""]


def test_resume_skips_done_and_upgrades_old_header(tmp_path):
    path = str(tmp_path / "m.csv")
    old_cols = [c for c in mamma_extraction._FIELDNAMES if c not in ("finish_reason", "completion_tokens")]
    pd.DataFrame([{c: "" for c in old_cols} | {"id": "m0", "parse_error": "False"}]).to_csv(path, index=False)
    client = FakeClient(_mamma_answer)
    mamma_extraction.run({}, client, _mamma_items(3), path)
    df = _read(path)
    assert list(df.columns[:len(mamma_extraction._FIELDNAMES)]) == mamma_extraction._FIELDNAMES
    assert sorted(df["id"]) == ["m0", "m1", "m2"]
    assert len(client.system_prompts) == 2           # m0 not asked again
    assert df.set_index("id").loc["m1", "completion_tokens"] == "101"


def test_unreadable_existing_csv_raises(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text('"id,benchmark\nm0,x\n', encoding="utf-8")   # unterminated quote
    with pytest.raises(RuntimeError, match="not readable"):
        mamma_extraction.run({}, FakeClient(_mamma_answer), _mamma_items(2), str(path))


def test_resume_without_id_column_raises(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        mamma_extraction.run({}, FakeClient(_mamma_answer), _mamma_items(2), str(path))


def test_server_unavailable_propagates_and_cancels(tmp_path):
    path = str(tmp_path / "m.csv")

    def answer(prompt):
        if _case(prompt) == 2:
            raise ServerUnavailableError("down")
        return _mamma_answer(prompt)

    client = FakeClient(answer)
    with pytest.raises(ServerUnavailableError):
        mamma_extraction.run({"benchmark_settings": {"concurrency": 1}}, client, _mamma_items(20), path)
    df = _read(path)
    assert {"m0", "m1"} <= set(df["id"])
    assert "m2" not in set(df["id"])
    assert len(df) == len(set(df["id"]))
    assert len(client.system_prompts) < 20           # pending requests were cancelled


def test_unexpected_worker_exception_does_not_run_remaining_items(tmp_path):
    path = str(tmp_path / "m.csv")

    def answer(prompt):
        if _case(prompt) == 1:
            raise KeyError("bug")
        return _mamma_answer(prompt)

    client = FakeClient(answer)
    with pytest.raises(KeyError):
        mamma_extraction.run({"benchmark_settings": {"concurrency": 1}}, client, _mamma_items(20), path)
    assert len(client.system_prompts) < 20


def test_max_errors_stops_early(tmp_path):
    path = str(tmp_path / "m.csv")
    client = FakeClient(lambda p: ("Error: HTTP 500", {}))
    cfg = {"benchmark_settings": {"max_errors": 3, "concurrency": 1}}
    mamma_extraction.run(cfg, client, _mamma_items(30), path)
    assert len(client.system_prompts) < 30
    assert len(_read(path)) <= 3


def test_format_rate_shows_seconds_per_item_when_slow():
    assert runner.format_rate(1, 12.5) == "12.5 s/item"
    assert runner.format_rate(30, 10) == "3.0 items/s"


# ---------------------------------------------------------------------------
# Arm task: system prompt, citation check inside workers, missing labels
# ---------------------------------------------------------------------------

_ARM_LABELS = ["Fracture", "Ossicles"]


def _arm_items(n):
    return [{"id": f"clavicle-{i}", "benchmark": "LabelExtractionArm",
             "text": f"Synthetisch: Fraktur der Clavicula Nummer {i}. CASE-{i}.",
             "region": "clavicle", "phase": "test",
             "gt_labels": {"Fracture": 1, "Ossicles": 0}, "template_labels": _ARM_LABELS}
            for i in range(n)]


def _arm_answer(prompt):
    i = _case(prompt)
    cite = f"Fraktur der Clavicula Nummer {i}" if i % 2 == 0 else "frei erfundenes Zitat"
    return json.dumps({"Fracture": {"finding": True, "citation": cite},
                       "Ossicles": {"finding": False, "citation": ""}}), {"finish_reason": "stop"}


def test_arm_uses_explicit_system_prompt_and_checks_citations_per_item(tmp_path):
    path = str(tmp_path / "a.csv")
    client = FakeClient(_arm_answer)
    arm_extraction.run({"benchmark_settings": {"concurrency": 4}}, client, _arm_items(20), path)
    assert set(client.system_prompts) == {arm_extraction._SYSTEM_PROMPT_ARM}
    assert "English" not in arm_extraction._SYSTEM_PROMPT_ARM
    df = _read(path)
    assert len(df) == 20
    for _, row in df.iterrows():
        i = int(row["id"].split("-")[1])
        assert json.loads(row["citation_check_json"]) == {"Fracture": i % 2 == 0}


def test_arm_parse_marks_null_finding_missing():
    parsed, err = arm_extraction._parse_response(
        json.dumps({"Fracture": {"finding": None, "citation": ""}, "Ossicles": False}), _ARM_LABELS)
    assert parsed["Fracture"]["finding"] is None and parsed["Ossicles"]["finding"] is False


# ---------------------------------------------------------------------------
# Arm evaluation
# ---------------------------------------------------------------------------

def _arm_row(rid, gt, pred, checks=None, parse_error=False, raw="", region="clavicle", finish=None):
    row = {"id": rid, "benchmark": "LabelExtractionArm", "region": region, "phase": "test",
           "gt_labels_json": json.dumps(gt), "model_raw": raw,
           "model_labels_json": json.dumps({k: {"finding": v, "citation": ""} for k, v in pred.items()}),
           "parse_error": str(parse_error), "citation_check_json": json.dumps(checks or {})}
    if finish is not None:
        row["finish_reason"] = finish
    return row


def _arm_eval(tmp_path, rows):
    p = tmp_path / "arm.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    r = write_arm_extraction_report_jsonl(str(p), str(tmp_path / "arm.jsonl"))
    lines = [json.loads(x) for x in open(tmp_path / "arm.jsonl", encoding="utf-8")]
    return r, [x for x in lines if x["type"] != "item"]


def test_arm_parse_error_not_scored_as_all_negative(tmp_path):
    gt = {"Fracture": 1, "Ossicles": 0}
    good = _arm_row("c1", gt, {"Fracture": True, "Ossicles": False})
    bad = _arm_row("c2", gt, {}, parse_error=True)
    r, _ = _arm_eval(tmp_path, [good, bad])
    assert r["n_parse_error"] == 1
    assert r["micro_f1_pct"] == 100.0          # the parse error adds no FN
    assert r["accuracy_pct"] == 100.0


def test_arm_missing_labels_counted_as_missing(tmp_path):
    gt = {"Fracture": 1, "Ossicles": 0}
    new_fmt = _arm_row("c1", gt, {"Fracture": None, "Ossicles": False})
    # old format: missing label stored as False, detected from the raw answer
    old_fmt = _arm_row("c2", gt, {"Fracture": False, "Ossicles": False},
                       raw=json.dumps({"Ossicles": {"finding": False, "citation": ""}}))
    r, rows = _arm_eval(tmp_path, [new_fmt, old_fmt])
    assert r["n_missing_labels"] == 2
    row = next(x for x in rows if x["metric"] == "n_missing_labels")
    assert row["n_missing_labels_gt_positive"] == 2
    # never scored as FN/negative: no Fracture decision enters the counts at all
    assert not [x for x in rows if x["type"] == "label_metric" and x["label"] == "Fracture"]
    oss = next(x for x in rows if x["type"] == "label_metric" and x["label"] == "Ossicles")
    assert oss["tn"] == 2


def test_arm_secondary_metrics_and_verbatim_split(tmp_path):
    gt = {"Fracture": 1, "Ossicles": 0, "Foreign Bodies": 0, "Lytic Lesion": 0}
    rows = [
        _arm_row("c1", gt, {"Fracture": True, "Ossicles": True, "Foreign Bodies": False, "Lytic Lesion": False},
                 checks={"Fracture": True, "Ossicles": False}, finish="stop"),
        _arm_row("c2", gt, {"Fracture": True, "Ossicles": False, "Foreign Bodies": False, "Lytic Lesion": False},
                 checks={"Fracture": True}, finish="length"),
    ]
    r, out = _arm_eval(tmp_path, rows)
    assert r["all_negative_baseline_accuracy_pct"] == 75.0
    assert r["accuracy_pct"] == 87.5
    assert r["verbatim_citation_rate_pct"] == r["citation_match_pct"] == round(2 / 3 * 100, 2)
    assert r["verbatim_rate_true_positive_pct"] == 100.0
    assert r["verbatim_rate_false_positive_pct"] == 0.0
    assert r["n_truncated"] == 1
    assert -1 <= r["mcc"] <= 1 and -1 <= r["clavicle_mcc"] <= 1
    by = {(x.get("region"), x["metric"]): x for x in out if x["type"] == "metric"}
    for key in ((None, "macro_f1_pct"), (None, "mcc"), ("clavicle", "micro_f1_pct"),
                ("clavicle", "macro_f1_pct")):
        assert by[key]["ci_lo"] <= by[key]["value"] <= by[key]["ci_hi"]
    assert by[(None, "accuracy_pct")]["secondary"] is True
    assert "deprecated" in by[(None, "citation_match_pct")]["note"]


def test_arm_mcc_perfect_and_inverse(tmp_path):
    gt = {"Fracture": 1, "Ossicles": 0}
    r, _ = _arm_eval(tmp_path, [_arm_row("c1", gt, {"Fracture": True, "Ossicles": False})])
    assert r["mcc"] == 1.0
    r, _ = _arm_eval(tmp_path, [_arm_row("c1", gt, {"Fracture": False, "Ossicles": True})])
    assert r["mcc"] == -1.0


# ---------------------------------------------------------------------------
# Mamma evaluation: sensitivity analyses, coverage, exam-level BPE, lesion CIs
# ---------------------------------------------------------------------------

def _mrow(i, **kw):
    row = {"id": f"x{i}", "benchmark": "LabelExtractionMamma",
           "gt_menopause": "", "gt_birads_li": "", "gt_birads_re": "",
           "gt_acr_li": "", "gt_acr_re": "", "gt_lesions_li": "[]", "gt_lesions_re": "[]",
           "model_raw": "", "model_menopause": "", "model_birads_li": "", "model_birads_re": "",
           "model_acr_li": "", "model_acr_re": "", "model_lesions_li": "[]", "model_lesions_re": "[]",
           "parse_error": "False"}
    row.update(kw)
    return row


def _mamma_eval(tmp_path, rows, config=None):
    p = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    r = write_mamma_extraction_report_jsonl(str(p), str(tmp_path / "m.jsonl"), config=config)
    lines = [json.loads(x) for x in open(tmp_path / "m.jsonl", encoding="utf-8")]
    return r, [x for x in lines if x["type"] == "metric"]


def test_birads6_sensitivity_rows_do_not_collide(tmp_path):
    rows = [_mrow(0, gt_birads_li="5", model_birads_li="6"),
            _mrow(1, gt_birads_li="2", model_birads_li="2"),
            _mrow(2, gt_birads_li="", model_birads_li="6")]
    r, out = _mamma_eval(tmp_path, rows)
    assert r["birads_li_accuracy_pct"] == 100.0             # primary: 6 -> 5
    assert r["birads_li_accuracy_birads6keep_pct"] == 50.0
    assert r["birads_li_n_model_birads6"] == 2
    primary = [x for x in out if x.get("field") == "birads_li" and x["metric"] == "accuracy_pct"]
    assert len(primary) == 1 and "variant" not in primary[0]
    keep = [x for x in out if x.get("variant") == "birads6_keep" and x["field"] == "birads_li"]
    assert {x["metric"] for x in keep} == {"accuracy_birads6keep_pct", "macro_f1_birads6keep_pct"}
    n6 = next(x for x in out if x["metric"] == "n_model_birads6" and x["field"] == "birads_li")
    assert n6["n_model_birads6_gt_present"] == 1


def test_birads6_variant_is_the_other_option(tmp_path):
    rows = [_mrow(0, gt_birads_li="5", model_birads_li="6")]
    cfg = {"task_settings": {"label_extraction_mamma": {"birads6_handling": "keep"}}}
    r, _ = _mamma_eval(tmp_path, rows, cfg)
    assert r["birads_li_accuracy_pct"] == 0.0
    assert r["birads_li_accuracy_birads6mapto5_pct"] == 100.0


def test_gt_empty_fp_sensitivity_and_ignored_counts(tmp_path):
    rows = [_mrow(0, gt_menopause="post", model_menopause="post",
                  gt_lesions_li=json.dumps(["ca"]), model_lesions_li=json.dumps(["invasives Karzinom"])),
            _mrow(1, gt_menopause="", model_menopause="prä",
                  model_lesions_li=json.dumps(["Zyste", "Narbe"]))]
    r, out = _mamma_eval(tmp_path, rows)
    assert r["menopause_accuracy_pct"] == 100.0
    assert r["menopause_accuracy_gtemptyfp_pct"] == 50.0
    assert r["menopause_n_ignored_model_values"] == 1
    assert r["lesions_li_micro_f1_pct"] == 100.0
    assert r["lesions_li_micro_f1_gtemptyfp_pct"] < 100.0
    ign = next(x for x in out if x["metric"] == "n_ignored_sides_model_present"
               and x["field"] == "lesions_li" and x["lesion_view"] == "type")
    assert ign["value"] == 1 and ign["n_ignored_model_entries"] == 2
    fp_rows = [x for x in out if x.get("variant") == "gt_empty_fp"]
    assert {x["field"] for x in fp_rows} == {"menopause", "birads_li", "birads_re", "acr_li",
                                             "acr_re", "lesions_li", "lesions_re"}


def test_coverage_explains_macro_above_accuracy(tmp_path):
    rows = [_mrow(i, gt_menopause="post", model_menopause="post" if i < 3 else "") for i in range(4)]
    r, out = _mamma_eval(tmp_path, rows)
    assert r["menopause_accuracy_pct"] == 75.0
    assert r["menopause_coverage_pct"] == 75.0
    assert r["menopause_macro_f1_pct"] > r["menopause_accuracy_pct"]
    macro = next(x for x in out if x["metric"] == "macro_f1_pct" and x["field"] == "menopause")
    assert "note" in macro


def test_acr_exam_level_accuracy(tmp_path):
    rows = [_mrow(0, gt_acr_li="1", gt_acr_re="1", model_acr_li="1", model_acr_re="1"),
            _mrow(1, gt_acr_li="2", gt_acr_re="2", model_acr_li="2", model_acr_re="3"),
            _mrow(2, gt_acr_li="1", gt_acr_re="2", model_acr_li="1", model_acr_re="2")]
    r, out = _mamma_eval(tmp_path, rows)
    assert r["acr_exam_accuracy_pct"] == 50.0
    row = next(x for x in out if x.get("field") == "acr_exam" and x["metric"] == "accuracy_pct")
    assert row["n_exams"] == 2 and row["n_exams_gt_li_ne_re"] == 1 and row["n_exams_model_li_ne_re"] == 1
    side = next(x for x in out if x["metric"] == "accuracy_pct" and x["field"] == "acr_li")
    assert "exam" in side["note"]


def test_lesion_precision_and_recall_have_cis(tmp_path):
    rows = [_mrow(i, gt_lesions_li=json.dumps(["ca", "Zyste"]),
                  model_lesions_li=json.dumps(["invasives Karzinom"] if i % 2 else ["Zyste", "Narbe"]))
            for i in range(10)]
    _, out = _mamma_eval(tmp_path, rows)
    for metric in ("micro_precision_pct", "micro_recall_pct", "micro_f1_pct"):
        row = next(x for x in out if x["metric"] == metric and x["field"] == "lesions_li"
                   and x["lesion_view"] == "type" and "variant" not in x)
        assert row["ci_lo"] <= row["value"] <= row["ci_hi"]


def test_mamma_n_truncated(tmp_path):
    rows = [_mrow(0, finish_reason="length", parse_error="True"), _mrow(1, finish_reason="stop")]
    r, out = _mamma_eval(tmp_path, rows)
    assert r["n_truncated"] == 1
    row = next(x for x in out if x["metric"] == "n_truncated")
    assert row["n_parse_error_truncated"] == 1


# ---------------------------------------------------------------------------
# Normalisation (D7)
# ---------------------------------------------------------------------------

def test_lymph_node_metastasis_not_benign_node():
    ln = _build_normalizer(_load_mamma_norm()["lesion_types"])
    assert ln["lymphknotenmetastase"] == "sonstige läsion"
    assert ln["lymphknoten"] == "lymphknoten"


def test_no_malignant_variant_maps_to_benign_type():
    ln = _build_normalizer(_load_mamma_norm()["lesion_types"])
    malignant_types = {"invasives karzinom", "dcis", "sonstige läsion"}
    for variant, canonical in ln.items():
        if any(t in variant for t in ("karzinom", "carcinom", "dcis", "metast", "malign")):
            assert canonical in malignant_types, variant


# ---------------------------------------------------------------------------
# CLI reads task_settings (D10)
# ---------------------------------------------------------------------------

def test_cli_reads_config(tmp_path, monkeypatch):
    import evaluate
    csv_path = tmp_path / "m.csv"
    pd.DataFrame([_mrow(0, gt_birads_li="5", model_birads_li="6")]).to_csv(csv_path, index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("task_settings:\n  label_extraction_mamma:\n    birads6_handling: keep\n", encoding="utf-8")
    out = tmp_path / "m.jsonl"
    monkeypatch.setattr(sys, "argv", ["evaluate.py", str(csv_path), "--type", "mamma_extraction",
                                      "--out", str(out), "--config", str(cfg)])
    evaluate.main()
    rows = [json.loads(x) for x in open(out, encoding="utf-8")]
    acc = next(x for x in rows if x.get("field") == "birads_li" and x.get("metric") == "accuracy_pct")
    assert acc["value"] == 0.0   # "6" kept as its own class -> wrong
