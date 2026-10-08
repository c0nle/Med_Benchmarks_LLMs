"""
Evaluation tests (client, letter parser, RadBench loader, Arm/Mamma metrics, judge, WBSS).
Synthetic data only – no patient data.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.client import MedicalLLMClient
from evaluate import (
    _build_normalizer,
    _categorical_metrics,
    _load_mamma_norm,
    extract_choice,
    score_vqa_mcq,
    write_arm_extraction_report_jsonl,
    write_mamma_extraction_report_jsonl,
)
from loaders.vision_benchmarks import _format_radbench_item
from tasks.arm_extraction import check_citations


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def test_empty_content_becomes_error():
    assert MedicalLLMClient._content_or_error(None, "length").startswith("Error: empty response")
    assert MedicalLLMClient._content_or_error("  ", "stop").startswith("Error:")
    assert MedicalLLMClient._content_or_error("A", "stop") == "A"


# ---------------------------------------------------------------------------
# MCQ letter extraction with >5 options
# ---------------------------------------------------------------------------

def test_extract_choice_extended_keys():
    keys = "ABCDEFGHIJKL"
    assert extract_choice("L", keys) == "L"
    assert extract_choice("The answer is K.", keys) == "K"
    assert extract_choice("F") is None  # default keys A-E


def test_score_vqa_mcq_twelve_options():
    options = [{"key": k, "value": f"T{i}"} for i, k in enumerate("ABCDEFGHIJKL", start=1)]
    df = pd.DataFrame([{
        "reference_answer": "T12", "model_answer": "L", "options_json": json.dumps(options),
    }])
    assert bool(score_vqa_mcq(df)["is_correct"].iloc[0])


# ---------------------------------------------------------------------------
# RadBench loader
# ---------------------------------------------------------------------------

def _radbench_row(**kw):
    row = {"CASE_ID": "case1", "A_TYPE": "CLOSED", "QUESTION": "q", "imageIDs": "", "_row": 7}
    row.update(kw)
    return row


def test_radbench_ranked_answer_takes_first():
    item = _format_radbench_item(_radbench_row(OPTIONS="femur, tibia, fibula", ANSWER="fibula,tibia,femur"), 0)
    assert item["answer"] == "fibula"


def test_radbench_plain_answer_unchanged():
    item = _format_radbench_item(_radbench_row(OPTIONS="left,right", ANSWER="right"), 0)
    assert item["answer"] == "right"


def test_radbench_id_unique_per_row():
    item = _format_radbench_item(_radbench_row(OPTIONS="yes,no", ANSWER="yes"), 0)
    assert item["id"] == "case1-q7"


def test_radbench_more_than_five_options_kept():
    opts = ",".join(f"T{i}" for i in range(1, 13))
    item = _format_radbench_item(_radbench_row(OPTIONS=opts, ANSWER="T12,T11"), 0)
    assert len(item["options"]) == 12
    assert item["answer"] == "T12"


# ---------------------------------------------------------------------------
# Arm citations
# ---------------------------------------------------------------------------

REPORT = "Fracture of the  middle third of the clavicle.\nNo dislocation of the AC joint."


def test_citation_verbatim_case_whitespace_insensitive():
    parsed = {"Fracture": {"finding": True, "citation": "fracture of the middle third"}}
    assert check_citations(parsed, REPORT) == {"Fracture": True}


def test_citation_fabricated_is_false():
    parsed = {"Fracture": {"finding": True, "citation": "displaced comminuted fracture"}}
    assert check_citations(parsed, REPORT) == {"Fracture": False}


def test_citation_fragments_with_ellipsis():
    parsed = {"Fracture": {"finding": True, "citation": "\"Fracture of the middle third ... AC joint\""}}
    assert check_citations(parsed, REPORT) == {"Fracture": True}


def test_citation_ignored_for_negative_or_empty():
    parsed = {
        "A": {"finding": False, "citation": "made up"},
        "B": {"finding": True, "citation": ""},
    }
    assert check_citations(parsed, REPORT) == {}


# ---------------------------------------------------------------------------
# Arm evaluation
# ---------------------------------------------------------------------------

def _arm_row(rid, region, gt, pred, checks):
    return {
        "id": rid, "benchmark": "LabelExtractionArm", "region": region, "phase": "test",
        "gt_labels_json": json.dumps(gt),
        "model_raw": "",
        "model_labels_json": json.dumps({k: {"finding": v, "citation": ""} for k, v in pred.items()}),
        "parse_error": "False",
        "citation_check_json": json.dumps(checks),
    }


def test_arm_eval_macro_excludes_undefined_and_splits_regions(tmp_path):
    rows = [
        _arm_row("c1", "clavicle", {"Fracture": 1, "Foreign Bodies": 0}, {"Fracture": True, "Foreign Bodies": False}, {"Fracture": True}),
        _arm_row("e1", "elbow", {"Fracture": 1, "Foreign Bodies": 1}, {"Fracture": False, "Foreign Bodies": True}, {"Foreign Bodies": False}),
    ]
    csv_path = tmp_path / "arm.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    r = write_arm_extraction_report_jsonl(str(csv_path), str(tmp_path / "arm.jsonl"))

    # clavicle "Foreign Bodies" has tp=fp=fn=0 → excluded; clavicle Fracture F1=1
    assert r["clavicle_macro_f1_pct"] == 100.0
    # elbow: Fracture F1=0, Foreign Bodies F1=1 → macro 50
    assert r["elbow_macro_f1_pct"] == 50.0
    # citation from task-side checks: 1 of 2 correct
    assert r["verbatim_citation_rate_pct"] == 50.0


def test_arm_eval_without_citation_column(tmp_path):
    row = _arm_row("c1", "clavicle", {"Fracture": 1}, {"Fracture": True}, {})
    del row["citation_check_json"]
    csv_path = tmp_path / "arm.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    r = write_arm_extraction_report_jsonl(str(csv_path), str(tmp_path / "arm.jsonl"))
    assert "verbatim_citation_rate_pct" not in r


# ---------------------------------------------------------------------------
# Mamma evaluation
# ---------------------------------------------------------------------------

def test_categorical_macro_ignores_missing_sentinel():
    m = _categorical_metrics(["2", "5"], ["2", "__missing__"])
    # class "2": F1=1, class "5": F1=0 → macro 0.5 (sentinel is not a class)
    assert abs(m["macro_f1"] - 0.5) < 1e-9


def test_mixed_lesions_normalize_to_dominant_type():
    ln = _build_normalizer(_load_mamma_norm()["lesion_types"])
    assert ln["ca/dcis"] == "invasives karzinom"
    assert ln["dcis/clis"] == "dcis"
    assert ln["unspezifisch"] == "unspezifische anreicherung"
    assert ln["mamille"] == "mamille"


def _mamma_row(gt_li, model_li):
    return {
        "id": "x", "benchmark": "LabelExtractionMamma",
        "gt_menopause": "post", "gt_birads_li": "", "gt_birads_re": "",
        "gt_acr_li": "", "gt_acr_re": "",
        "gt_lesions_li": json.dumps(gt_li), "gt_lesions_re": "[]",
        "model_raw": "", "model_menopause": "post",
        "model_birads_li": "", "model_birads_re": "", "model_acr_li": "", "model_acr_re": "",
        "model_lesions_li": json.dumps(model_li), "model_lesions_re": "[]",
        "parse_error": "False",
    }


def test_mamma_multifocal_type_vs_count(tmp_path):
    # GT: 4 carcinoma foci + DCIS; model names each type once
    csv_path = tmp_path / "m.csv"
    pd.DataFrame([_mamma_row(["ca"] * 4 + ["dcis"], ["invasives Karzinom", "DCIS"])]).to_csv(csv_path, index=False)
    r = write_mamma_extraction_report_jsonl(str(csv_path), str(tmp_path / "m.jsonl"))
    assert r["lesions_li_micro_f1_pct"] == 100.0          # main: types match
    assert r["lesions_li_exact_match_pct"] == 100.0
    assert r["lesions_li_count_micro_f1_pct"] == 57.14    # tp=2, fn=3 → 4/7
    assert r["lesions_li_count_exact_match_pct"] == 0.0


def test_mamma_exact_match_order_insensitive(tmp_path):
    csv_path = tmp_path / "m.csv"
    pd.DataFrame([_mamma_row(["DCIS", "ca"], ["invasives Karzinom", "DCIS"])]).to_csv(csv_path, index=False)
    r = write_mamma_extraction_report_jsonl(str(csv_path), str(tmp_path / "m.jsonl"))
    assert r["lesions_li_exact_match_pct"] == 100.0
    assert r["lesions_li_micro_f1_pct"] == 100.0


# ---------------------------------------------------------------------------
# Letter parser, judge, yes/no, WBSS, robustness
# ---------------------------------------------------------------------------

from evaluate import (  # noqa: E402
    _wbss,
    _yes_no_token,
    parse_judge_reply,
)
from tasks.arm_extraction import _parse_response as arm_parse  # noqa: E402
from tasks.mamma_extraction import _parse_response as mamma_parse  # noqa: E402


def test_extract_choice_ignores_pronoun_and_article():
    assert extract_choice("I think the answer is B", "ABCDEFGHIJKL") == "B"
    assert extract_choice("I'd choose B") == "B"
    assert extract_choice("This is a fracture; answer: C") == "C"
    assert extract_choice("**B**") == "B"
    assert extract_choice("(D)") == "D"
    assert extract_choice("Error: timeout") is None


def test_parse_judge_reply_strict():
    assert parse_judge_reply("1") == 1
    assert parse_judge_reply(" **0** ") == 0
    assert parse_judge_reply("<think>reply 1 or 0</think>0") == 0
    assert parse_judge_reply("The answer is 1 because...") is None
    assert parse_judge_reply("Error: empty response") is None


def test_yes_no_first_token():
    assert _yes_no_token("Yes, there is a fracture.") == "yes"
    assert _yes_no_token("No.") == "no"
    assert _yes_no_token("Possibly") is None


def test_wbss_identical_non_wordnet_tokens():
    assert _wbss("t2", "t2") == 1.0
    assert _wbss("CTA - CT angiography", "cta - ct angiography") == 1.0


def test_arm_string_false_is_negative_and_non_dict_is_error():
    labels = ["Fracture", "Ossicles"]
    parsed, err = arm_parse(json.dumps({"fracture": {"finding": "false", "citation": ""},
                                        "Ossicles": {"finding": "true", "citation": "x"}}), labels)
    assert err is False
    assert parsed["Fracture"]["finding"] is False
    assert parsed["Ossicles"]["finding"] is True
    assert arm_parse("[1, 2]", labels)[1] is True
    assert arm_parse("null", labels)[1] is True


def test_arm_unwraps_nested_findings():
    labels = ["Fracture", "Ossicles"]
    raw = json.dumps({"findings": {"Fracture": {"finding": True, "citation": "x"},
                                   "Ossicles": {"finding": False, "citation": ""}}})
    parsed, err = arm_parse(raw, labels)
    assert err is False and parsed["Fracture"]["finding"] is True


def test_citation_too_short_fails():
    assert check_citations({"A": {"finding": True, "citation": "a"}}, REPORT) == {"A": False}


def test_mamma_non_dict_json_is_parse_error():
    assert mamma_parse("[1, 2]")[1] is True
    assert mamma_parse("null")[1] is True


def test_mamma_lesions_on_gt_empty_side_ignored(tmp_path):
    csv_path = tmp_path / "m.csv"
    rows = [_mamma_row([], ["Hämatom"]), _mamma_row(["ca"], ["invasives Karzinom"])]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    r = write_mamma_extraction_report_jsonl(str(csv_path), str(tmp_path / "m.jsonl"))
    assert r["lesions_li_micro_f1_pct"] == 100.0
    cfg = {"task_settings": {"label_extraction_mamma": {"gt_empty_ext_present": "fp"}}}
    r_fp = write_mamma_extraction_report_jsonl(str(csv_path), str(tmp_path / "m2.jsonl"), config=cfg)
    assert r_fp["lesions_li_micro_f1_pct"] < 100.0


def test_client_retries_transient_then_gives_up():
    cfg = {"server": {"url": "http://localhost:1/v1", "model_name": "m", "client": "requests",
                      "max_retries": 2, "retry_backoff_s": 0, "max_consecutive_errors": 100}}
    c = MedicalLLMClient(cfg)
    calls = []
    c._call_requests = lambda m: calls.append(1) or "Error: HTTP 503 down"
    assert c.ask_question("x").startswith("Error:")
    assert len(calls) == 3
    calls.clear()
    c._call_requests = lambda m: calls.append(1) or "Error: HTTP 400 bad"
    c.ask_question("x")
    assert len(calls) == 1


def test_client_stops_after_consecutive_failures():
    import pytest
    from core.client import ServerUnavailableError
    cfg = {"server": {"url": "http://localhost:1/v1", "model_name": "m", "client": "requests",
                      "max_retries": 0, "max_consecutive_errors": 3}}
    c = MedicalLLMClient(cfg)
    c._call_requests = lambda m: "Error: timed out"
    c.ask_question("x"); c.ask_question("x")
    with pytest.raises(ServerUnavailableError):
        c.ask_question("x")


def test_menopause_in_report_detection():
    from loaders.mamma_extraction import menopause_in_report
    assert menopause_in_report("Indikation: Staging. Postmenopausal.")
    assert menopause_in_report("Untersuchung am 9. Zyklustag.")
    assert menopause_in_report("Patientin prämenopausal, ZT 12")
    assert not menopause_in_report("Indikation: Staging bei Mammakarzinom rechts.")


def test_mamma_menopause_accuracy_in_report(tmp_path):
    # 3 exams with GT "post": stated + right, stated + missing, not stated + missing
    rows = []
    for rid, in_report, model in (("a", "True", "post"), ("b", "True", ""), ("c", "False", "")):
        r = _mamma_row([], [])
        r.update({"id": rid, "menopause_in_report": in_report, "model_menopause": model})
        rows.append(r)
    csv_path = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    r = write_mamma_extraction_report_jsonl(str(csv_path), str(tmp_path / "m.jsonl"))
    assert r["menopause_accuracy_pct"] == round(100 / 3, 2)      # all 3 exams
    assert r["menopause_accuracy_in_report_pct"] == 50.0         # only a and b
    row = next(json.loads(l) for l in open(tmp_path / "m.jsonl")
               if '"accuracy_in_report_pct"' in l)
    assert row["n"] == 2 and row["n_not_in_report"] == 1


def test_mamma_menopause_in_report_absent_column(tmp_path):
    csv_path = tmp_path / "m.csv"
    pd.DataFrame([_mamma_row([], [])]).to_csv(csv_path, index=False)
    r = write_mamma_extraction_report_jsonl(str(csv_path), str(tmp_path / "m.jsonl"))
    assert "menopause_accuracy_in_report_pct" not in r and r["menopause_accuracy_pct"] == 100.0
