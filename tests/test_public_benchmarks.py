"""
Tests for the public-benchmark pipeline (MCQ / VQA tasks, loaders, evaluation).
Synthetic data and a fake client only – no server calls, no patient data.
"""
import json
import os
import sys
import threading

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluate as ev  # noqa: E402
from core.client import ServerUnavailableError  # noqa: E402
from evaluate import (  # noqa: E402
    _exact_match,
    _judge_cache_path,
    cluster_bootstrap_ci,
    evaluate_vqa_with_judge,
    extract_choice,
    judge_prompt,
    score_vqa_mcq,
    wilson_ci,
    write_report_jsonl,
    write_vqa_report_jsonl,
)


# ---------------------------------------------------------------------------
# Fake OpenAI-compatible client (interface of core.client.MedicalLLMClient)
# ---------------------------------------------------------------------------

class FakeClient:
    def __init__(self, reply=None, model="fake/judge-1"):
        self.model = model
        self.reply = reply or (lambda prompt: "A")
        self.calls = []
        self._lock = threading.Lock()
        self._local = threading.local()

    @property
    def last_meta(self):
        return getattr(self._local, "meta", None)

    def _answer(self, prompt):
        with self._lock:
            self.calls.append(prompt)
        ans = self.reply(prompt)
        self._local.meta = {"finish_reason": "length" if "LONG" in prompt else "stop",
                            "completion_tokens": 7, "model": self.model}
        return ans

    def ask_question(self, prompt, system_prompt=None):
        return self._answer(prompt)

    def ask_with_images(self, prompt, images_b64, image_format="jpeg", system_prompt=None):
        assert images_b64
        return self._answer(prompt)


def _read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _metrics(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if '"type": "metric"' in line]


def _metric(rows, metric, subset=None):
    hits = [r for r in rows if r["metric"] == metric and r.get("subset") == subset]
    assert len(hits) == 1, (metric, subset, hits)
    return hits[0]


# ---------------------------------------------------------------------------
# MCQ letter extraction (real cases from run_full_20261002)
# ---------------------------------------------------------------------------

def test_last_explicit_answer_wins_self_correction():
    # MedQA 5432f150…: "The correct answer is **B**." … "**Correct Letter: C**"
    text = ("The correct answer is **B**.\n\n**Explanation:** … **Option C** describes Neuroblastoma.\n\n"
            "*Correction/Refinement:* Upon closer review, **Option C** is the definitive description.\n\n"
            "**Correct Letter: C** ")
    assert extract_choice(text, "ABCD") == "C"


def test_curve_list_does_not_give_out_of_range_letter():
    # MedQA e006336d… (4 options): "- **Curve C/D/E** …" must not become E
    text = ("- **Curve A** usually represents glucose.\n- **Curve B** represents Na.\n"
            "- **Curve C/D/E** (depending on the specific graph) represents poorly reabsorbed substances.")
    assert extract_choice(text, "ABCD") is None


@pytest.mark.parametrize("text,keys,expected", [
    ("A and B", "ABCD", None),
    ("Both A and C are correct.", "ABCD", None),
    ("The answer is A, C", "ABCD", None),
    ("Answer: d", "ABCD", "D"),
    ("The best answer: (c)", "ABCD", "C"),
    ("The answer is a fracture of the radius.", "ABCD", None),
    ("Answer: E", "ABCD", None),                       # outside the option set
    ("E", "ABCD", None),
    ("None of the provided options (A, B, C, D) are correct.", "ABCD", None),
    ("**Answer:** C", "ABCD", "C"),
    ("The correct answer is **(B)**.", "ABCD", "B"),
    ("B. Pneumothorax", "ABCD", "B"),
    ("The answer is B and a CT is recommended.", "ABCD", "B"),
    ("<think>maybe A</think>C", "ABCD", "C"),
])
def test_extract_choice_cases(text, keys, expected):
    assert extract_choice(text, keys) == expected


def test_mcq_report_uses_option_keys_and_wilson_ci(tmp_path):
    rows = [
        {"id": "1", "benchmark": "MedQA", "question": "q", "correct_answer": "D", "option_keys": "ABCD",
         "model_answer": "Answer: d", "finish_reason": "stop", "completion_tokens": "3"},
        {"id": "2", "benchmark": "MedQA", "question": "q", "correct_answer": "A", "option_keys": "ABCD",
         "model_answer": "Curve C/D/E", "finish_reason": "length", "completion_tokens": "256"},
        {"id": "3", "benchmark": "MedQA", "question": "q", "correct_answer": "B", "option_keys": "ABCD",
         "model_answer": "None", "finish_reason": "stop", "completion_tokens": "1"},
    ]
    csv_path = tmp_path / "medqa_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    r = write_report_jsonl(str(csv_path), str(tmp_path / "medqa_report.jsonl"))
    assert r["accuracy_pct"] == pytest.approx(33.33)
    m = _metrics(tmp_path / "medqa_report.jsonl")
    acc = _metric(m, "accuracy_pct")
    assert acc["ci_method"] == "wilson" and acc["n"] == 3
    lo, hi = wilson_ci(1, 3)
    assert acc["ci_lo"] == round(lo * 100, 2) and acc["ci_hi"] == round(hi * 100, 2)
    assert _metric(m, "n_truncated")["value"] == 1


def test_csv_answer_none_is_text_not_nan(tmp_path):
    csv_path = tmp_path / "x_results.csv"
    pd.DataFrame([{"id": "1", "question_type": "open", "question": "q", "reference_answer": "none",
                   "model_answer": "None"}]).to_csv(csv_path, index=False)
    r = write_vqa_report_jsonl(str(csv_path), str(tmp_path / "x.jsonl"))
    assert r["open_exact_match_pct"] == 100.0


# ---------------------------------------------------------------------------
# VQA scoring
# ---------------------------------------------------------------------------

def test_vqa_mcq_letter_outside_options_is_wrong():
    options = [{"key": k, "value": v} for k, v in zip("AB", ["left", "right"])]
    df = pd.DataFrame([
        {"reference_answer": "right", "model_answer": "C", "options_json": json.dumps(options)},
        {"reference_answer": "right", "model_answer": "B", "options_json": json.dumps(options)},
        {"reference_answer": "right", "model_answer": "right", "options_json": json.dumps(options)},
    ])
    assert score_vqa_mcq(df)["is_correct"].tolist() == [False, True, True]


def test_exact_match_any_alternative():
    assert _exact_match("CT w/ contrast IV", ["ct w/contrast", "ct w/contrast iv"])
    assert not _exact_match("mri", ["ct w/contrast", "ct w/contrast iv"])


def test_judge_prompt_shows_all_references_and_rubric():
    p = judge_prompt("what modality?", ["ct w/contrast", "ct w/contrast iv"], "CT")
    assert "Reference answer(s): ct w/contrast | ct w/contrast iv" in p
    assert "more specific" in p and "'1'" in p


def test_wilson_and_cluster_bootstrap():
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4) and hi == pytest.approx(0.9433, abs=1e-4)
    vals = [1, 1, 1, 0, 0, 0, 1, 0]
    clusters = ["a", "a", "a", "b", "b", "b", "c", "c"]
    ci1 = cluster_bootstrap_ci(vals, clusters)
    assert ci1 == cluster_bootstrap_ci(vals, clusters)       # seed 42, deterministic
    assert ci1[0] <= 0.5 <= ci1[1]


def _vqa_rows():
    rows = []
    for i in range(12):
        cat = "plane" if i < 6 else "organ"
        refs = ["axial", "transverse"] if i == 0 else ["axial" if cat == "plane" else "lung"]
        rows.append({
            "id": f"q{i}", "benchmark": "VQA-Med-2019", "question_type": "open", "category": cat,
            "cluster_id": f"img{i // 2}", "question": f"question {i}", "reference_answer": refs[0],
            "reference_answers_json": json.dumps(refs),
            "model_answer": ("transverse" if i == 0 else ("axial" if i % 3 else "coronal"))
            if cat == "plane" else ("lung" if i % 2 else "Error: timeout"),
            "options_json": "[]", "finish_reason": "stop", "completion_tokens": "2",
        })
    return rows


def test_vqa_report_categories_ci_and_shuffled_baseline(tmp_path):
    csv_path = tmp_path / "vqa_med_2019_results.csv"
    pd.DataFrame(_vqa_rows()).to_csv(csv_path, index=False)
    r = write_vqa_report_jsonl(str(csv_path), str(tmp_path / "r.jsonl"))
    m = _metrics(tmp_path / "r.jsonl")
    em = _metric(m, "exact_match_pct", "open")
    assert em["ci_method"] == "cluster_bootstrap" and em["n_clusters"] == 6
    assert "ci_lo_wilson" in em and em["ci_lo"] <= em["value"] <= em["ci_hi"]
    plane = _metric(m, "exact_match_pct", "open:category=plane")
    assert plane["n"] == 6
    # q0 matches the second alternative; q3 answers "coronal" → 5/6
    assert plane["value"] == pytest.approx(83.33)
    assert "open_wbss_shuffled_baseline_pct" in r
    _metric(m, "wbss_shuffled_baseline_pct", "open")


# ---------------------------------------------------------------------------
# LLM-as-a-Judge
# ---------------------------------------------------------------------------

def test_judge_cache_unparsed_errors_and_model_slug(tmp_path):
    results_csv = tmp_path / "vqa_med_2019_results.csv"
    pd.DataFrame(_vqa_rows()).to_csv(results_csv, index=False)

    def reply(prompt):
        if "question 1\n" in prompt:
            return "The answer is correct, so 1"     # unparsed → wrong
        if "question 2\n" in prompt:
            return "Error: HTTP 500"                   # judge error → wrong
        return "1"

    judge = FakeClient(reply=reply, model="org/judge-A")
    r = write_vqa_report_jsonl(str(results_csv), str(tmp_path / "r.jsonl"), client=judge, run_judge=True)
    cache = tmp_path / "vqa_med_2019_judge_cache_org_judge-A.csv"
    assert cache.exists()
    assert str(cache) == _judge_cache_path(str(results_csv), "org/judge-A")
    # 12 rows: 3 model errors (not judged), 1 unparsed, 1 judge error → 7 correct
    assert r["open_judge_accuracy_pct"] == pytest.approx(7 / 12 * 100, abs=0.01)
    assert r["open_n_judge_unparsed"] == 1 and r["open_n_judge_errors"] == 1
    assert r["judge_model"] == "org/judge-A"
    m = _metrics(tmp_path / "r.jsonl")
    row = _metric(m, "llm_judge_accuracy_pct", "open")
    assert row["judge_model"] == "org/judge-A" and row["n_judge_errors"] == 1
    assert len(judge.calls) == 9

    # Second evaluation: parsed verdicts come from the cache, unparsed/error are asked again
    judge2 = FakeClient(reply=reply, model="org/judge-A")
    write_vqa_report_jsonl(str(results_csv), str(tmp_path / "r2.jsonl"), client=judge2, run_judge=True)
    assert len(judge2.calls) == 2

    # Another judge model never reuses these verdicts
    judge3 = FakeClient(reply=lambda p: "0", model="other-judge")
    r3 = write_vqa_report_jsonl(str(results_csv), str(tmp_path / "r3.jsonl"), client=judge3, run_judge=True)
    assert len(judge3.calls) == 9 and r3["open_judge_accuracy_pct"] == 0.0


def test_judge_cache_key_changes_with_answer(tmp_path):
    df = pd.DataFrame(_vqa_rows()[:2])
    cache = str(tmp_path / "c.csv")
    j = FakeClient(reply=lambda p: "1")
    evaluate_vqa_with_judge(df, j, cache_path=cache, workers=2)
    df.loc[0, "model_answer"] = "sagittal"
    j2 = FakeClient(reply=lambda p: "0")
    out = evaluate_vqa_with_judge(df, j2, cache_path=cache, workers=2)
    assert len(j2.calls) == 1
    assert out["judge_correct"].tolist() == [0, 1]


def test_legacy_judge_cache_is_ignored(tmp_path):
    results_csv = tmp_path / "radbench_results.csv"
    pd.DataFrame(_vqa_rows()[:3]).to_csv(results_csv, index=False)
    pd.DataFrame([{"id": "q0", "judge_raw": "0"}]).to_csv(tmp_path / "radbench_judge_cache.csv", index=False)
    j = FakeClient(reply=lambda p: "1")
    r = write_vqa_report_jsonl(str(results_csv), str(tmp_path / "r.jsonl"), client=j, run_judge=True)
    assert len(j.calls) == 3 and r["open_judge_accuracy_pct"] == 100.0


def test_judge_server_unavailable_propagates(tmp_path):
    def reply(prompt):
        raise ServerUnavailableError("down")
    with pytest.raises(ServerUnavailableError):
        evaluate_vqa_with_judge(pd.DataFrame(_vqa_rows()[:4]), FakeClient(reply=reply),
                                cache_path=str(tmp_path / "c.csv"), workers=2)


# ---------------------------------------------------------------------------
# Task runners: concurrency, resume, meta columns
# ---------------------------------------------------------------------------

def _mcq_items(n):
    return [{"id": f"m{i}", "benchmark": "MedQA", "question": f"Q{i}" + (" LONG" if i == 0 else ""),
             "options": [{"key": k, "value": k.lower()} for k in "ABCD"], "correct_answer": "A"}
            for i in range(n)]


def test_mcq_task_concurrent_writes_all_rows_with_meta(tmp_path):
    from tasks import mcq
    path = str(tmp_path / "medqa_results.csv")
    client = FakeClient(reply=lambda p: "A")
    cfg = {"benchmark_settings": {"concurrency": 4}}
    mcq.run(cfg, client, _mcq_items(30), path)
    df = _read(path)
    assert sorted(df["id"]) == sorted(f"m{i}" for i in range(30))
    assert set(df["option_keys"]) == {"ABCD"}
    assert set(df["completion_tokens"]) == {"7"}
    assert df.loc[df["id"] == "m0", "finish_reason"].item() == "length"
    assert (df.loc[df["id"] != "m0", "finish_reason"] == "stop").all()
    r = write_report_jsonl(path, str(tmp_path / "medqa_report.jsonl"))
    assert r["accuracy_pct"] == 100.0 and r["n_truncated"] == 1


def test_resume_skips_done_and_migrates_old_header(tmp_path):
    from tasks import mcq
    path = tmp_path / "medqa_results.csv"
    pd.DataFrame([{"id": "m0", "benchmark": "MedQA", "question": "Q0", "correct_answer": "A",
                   "model_answer": "None"}]).to_csv(path, index=False)
    client = FakeClient(reply=lambda p: "B")
    mcq.run({"benchmark_settings": {"concurrency": 2}}, client, _mcq_items(3), str(path))
    df = _read(path)
    assert len(client.calls) == 2 and len(df) == 3
    assert df.loc[df["id"] == "m0", "model_answer"].item() == "None"
    assert list(df.columns)[:5] == ["id", "benchmark", "question", "correct_answer", "model_answer"]
    assert {"option_keys", "finish_reason", "completion_tokens"} <= set(df.columns)


def test_unreadable_results_file_raises(tmp_path):
    from tasks import mcq
    path = tmp_path / "medqa_results.csv"
    path.write_bytes(b"\x00\x01garbage-without-id\n\"unterminated")
    with pytest.raises(RuntimeError):
        mcq.run({}, FakeClient(), _mcq_items(2), str(path))


def test_server_unavailable_stops_benchmark(tmp_path):
    from tasks import mcq
    calls = []

    def reply(prompt):
        calls.append(prompt)
        if len(calls) >= 3:
            raise ServerUnavailableError("down")
        return "A"

    path = str(tmp_path / "medqa_results.csv")
    with pytest.raises(ServerUnavailableError):
        mcq.run({"benchmark_settings": {"concurrency": 1}}, FakeClient(reply=reply), _mcq_items(50), path)
    assert len(_read(path)) == 2
    assert len(calls) < 10


def test_max_errors_stops_submitting(tmp_path):
    from tasks import mcq
    client = FakeClient(reply=lambda p: "Error: HTTP 400 bad")
    path = str(tmp_path / "medqa_results.csv")
    mcq.run({"benchmark_settings": {"concurrency": 2, "max_errors": 3}}, client, _mcq_items(40), path)
    assert 3 <= len(_read(path)) <= 3 + 4


def test_vqa_task_rows_and_open_prompt(tmp_path):
    from PIL import Image
    from tasks import vqa
    img = Image.new("RGB", (8, 8))
    items = [
        {"id": "v0", "benchmark": "VQA-Med-2019", "question": "which plane?", "answer": "axial",
         "answers": ["axial", "transverse"], "image": img, "image_format": "jpeg",
         "meta": {"category": "plane", "cluster_id": "synpic1.jpg"}},
        {"id": "v1", "benchmark": "RadBench", "question": "pneumothorax?", "answer": "yes",
         "image": img, "image_format": "jpeg",
         "meta": {"question_type": "yes_no", "category": "pathology", "cluster_id": "77654",
                  "all_images": [img, img]}},
    ]
    client = FakeClient(reply=lambda p: "axial")
    path = str(tmp_path / "x_results.csv")
    vqa.run({"benchmark_settings": {"concurrency": 2}}, client, items, path)
    df = _read(path).set_index("id")
    assert json.loads(df.loc["v0", "reference_answers_json"]) == ["axial", "transverse"]
    assert df.loc["v0", "category"] == "plane" and df.loc["v1", "cluster_id"] == "77654"
    open_prompt = [p for p in client.calls if "which plane" in p][0]
    assert open_prompt.endswith("Answer the question with a single word or a short phrase.")
    assert "key medical terms" not in open_prompt


def test_vqa_task_refuses_unencodable_image(tmp_path):
    from tasks import vqa
    item = {"id": "v0", "question": "q", "answer": "a", "image": None, "image_format": "jpeg",
            "meta": {"all_images": ["not an image"]}}
    with pytest.raises(ValueError):
        vqa.run({}, FakeClient(), [item], str(tmp_path / "x_results.csv"))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def test_radbench_image_filenames_unique_for_same_last_segment():
    from loaders.vision_benchmarks import radbench_image_filename
    urls = [f"https://prod-images-static.radiopaedia.org/images/{n}/0._jumbo.jpeg"
            for n in (58649157, 58755746, 58077143, 58750692)]
    names = {radbench_image_filename(u) for u in urls}
    assert len(names) == 4 and all(n.endswith(".jpeg") for n in names)
    assert radbench_image_filename(urls[0]) == radbench_image_filename(" " + urls[0] + " ")


def _radbench_fixture(tmp_path, monkeypatch, cache_refs):
    from PIL import Image
    import loaders.vision_benchmarks as vb
    u1 = "https://x.org/images/1/0._jumbo.jpeg"
    u2 = "https://x.org/images/2/0._jumbo.jpeg"
    rows = [
        {"imageSource": "radiopaedia", "CASE_ID": 10, "imageIDs": f"{u1},{u2}", "QUESTION": "compare",
         "Q_TYPE": "comparison", "ANSWER": "yes", "A_TYPE": "CLOSED", "OPTIONS": "yes,no"},
        {"imageSource": "radiopaedia", "CASE_ID": 11, "imageIDs": f"{u1}, 52662257", "QUESTION": "compare",
         "Q_TYPE": "comparison", "ANSWER": "drain", "A_TYPE": "OPEN", "OPTIONS": ""},
        {"imageSource": "medpix", "CASE_ID": "", "imageIDs": "01ba570d-e65c-47a2-9abf-acc74b57351a",
         "QUESTION": "q", "Q_TYPE": "anatomy", "ANSWER": "x", "A_TYPE": "OPEN", "OPTIONS": ""},
    ]
    csv_path = tmp_path / "radbench.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    monkeypatch.setattr(vb, "_RADBENCH_PATH", csv_path)
    monkeypatch.setattr(vb, "_RADBENCH_IMAGE_DIR", img_dir)
    for i, ref in enumerate(cache_refs):
        Image.new("RGB", (4 + i, 4)).save(vb.radbench_image_path({"u1": u1, "u2": u2}[ref]))
    return vb


def test_radbench_loader_sends_all_distinct_images(tmp_path, monkeypatch):
    vb = _radbench_fixture(tmp_path, monkeypatch, ["u1", "u2"])
    items = vb.load_radbench()
    # MedPix row and the bare-id row are dropped (reported), the URL row keeps both images
    assert [it["id"] for it in items] == ["10.0-q0"]
    imgs = items[0]["meta"]["all_images"]
    assert len(imgs) == 2 and imgs[0].size != imgs[1].size
    assert items[0]["meta"]["cluster_id"] == "10" and items[0]["meta"]["category"] == "comparison"


def test_radbench_missing_image_raises_by_default(tmp_path, monkeypatch):
    vb = _radbench_fixture(tmp_path, monkeypatch, ["u1"])
    with pytest.raises(FileNotFoundError):
        vb.load_radbench()
    cfg = {"task_settings": {"radbench": {"missing_images": "skip_question"}}}
    assert vb.load_radbench(config=cfg) == []


def test_vqa_med_keeps_all_answers_and_category():
    from loaders.vision_benchmarks import _format_vqa_med_item
    it = _format_vqa_med_item({"question": "q", "answer": ["ct w/contrast", "ct w/contrast iv"],
                               "question_categories": "modality", "_image_path": "synpic1.jpg"}, 3)
    assert it["answer"] == "ct w/contrast" and it["answers"] == ["ct w/contrast", "ct w/contrast iv"]
    assert it["meta"]["category"] == "modality" and it["meta"]["cluster_id"] == "synpic1.jpg"
    it2 = _format_vqa_med_item({"question": "q", "answer": "['a', 'b']"}, 0)
    assert it2["answers"] == ["a", "b"]


def test_radimagenet_category_and_cluster():
    from loaders.vision_benchmarks import _format_radimagenet_benchmark_item
    it = _format_radimagenet_benchmark_item({
        "question": "q", "answer": "lung", "question_type": "open", "_image_path": "lung1.png",
        "metadata": {"content_type": "anatomy", "question_id": "anatomy_open", "modality": "ct"}}, 5)
    assert it["id"] == "anatomy_open-5"
    assert it["meta"]["category"] == "anatomy" and it["meta"]["cluster_id"] == "lung1.png"
