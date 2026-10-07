"""
Pipeline tests (run dir handling, client, summary, plot). Synthetic data only –
no patient data, no network access (HTTP is mocked).
"""
import json
import os
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
import requests
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as main_mod                                                 # noqa: E402
from core import run_utils                                              # noqa: E402
from core.client import (ConfigurationError, MedicalLLMClient,          # noqa: E402
                         ServerUnavailableError, DEFAULT_SYSTEM_PROMPT)
from core.logger import RunLogger                                       # noqa: E402
from core.summary import flatten_report_rows, merge_metrics, format_metrics_line  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)


def _ok_payload(content="A", model="served-model", finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 2}, "model": model}


def _cfg(**server):
    base = {"url": "http://localhost:1/v1", "model_name": "m", "client": "requests",
            "max_retries": 0, "retry_backoff_s": 0, "max_consecutive_errors": 1000}
    base.update(server)
    return {"server": base, "benchmark_settings": {"temperature": 0, "max_tokens": 16}}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def test_last_meta_seed_and_extra_body(monkeypatch):
    sent = {}

    def fake_post(url, headers, json, timeout, verify):
        sent.update(json)
        return _FakeResponse(200, _ok_payload())

    monkeypatch.setattr(requests, "post", fake_post)
    c = MedicalLLMClient(_cfg(seed=7, extra_body={"chat_template_kwargs": {"enable_thinking": False}}))
    assert c.ask_question("q") == "A"
    meta = c.last_meta
    assert meta["finish_reason"] == "stop"
    assert meta["prompt_tokens"] == 11 and meta["completion_tokens"] == 2
    assert meta["model"] == "served-model"
    assert isinstance(meta["latency_s"], float)
    assert sent["seed"] == 7
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["messages"][0]["content"] == DEFAULT_SYSTEM_PROMPT
    assert c.reported_models == {"served-model"}

    c.ask_with_image("q", "aGk=", system_prompt="custom")
    assert sent["messages"][0]["content"] == "custom"
    assert sent["messages"][1]["content"][0]["type"] == "image_url"


def test_last_meta_on_error(monkeypatch):
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _FakeResponse(200, _ok_payload(content="", finish="length")))
    c = MedicalLLMClient(_cfg())
    assert c.ask_question("q").startswith("Error: empty response")
    assert c.last_meta["finish_reason"] == "error"
    assert c.last_meta["server_finish_reason"] == "length"


def test_last_meta_is_per_thread(monkeypatch):
    def fake_post(url, headers, json, timeout, verify):
        prompt = json["messages"][1]["content"]
        return _FakeResponse(200, _ok_payload(content=prompt, model=f"model-{prompt}"))

    monkeypatch.setattr(requests, "post", fake_post)
    c = MedicalLLMClient(_cfg())

    def work(i):
        ans = c.ask_question(str(i))
        return ans, c.last_meta["model"]

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(work, range(200)))
    assert all(model == f"model-{ans}" for ans, model in results)


def test_error_counter_is_thread_safe():
    c = MedicalLLMClient(_cfg(max_consecutive_errors=10_000))
    c._call_requests = lambda m: "Error: timed out"
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: c.ask_question("x"), range(400)))
    assert c._consecutive_errors == 400

    c2 = MedicalLLMClient(_cfg(max_consecutive_errors=50))
    c2._call_requests = lambda m: "Error: timed out"
    raised = []

    def call(_):
        try:
            c2.ask_question("x")
        except ServerUnavailableError:
            raised.append(1)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(call, range(100)))
    assert len(raised) == 51          # calls 50..100 all see a counter >= 50


@pytest.mark.parametrize("status", [401, 403, 404])
def test_auth_errors_abort_immediately(monkeypatch, status):
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        return _FakeResponse(status, {"error": "nope"})

    monkeypatch.setattr(requests, "post", fake_post)
    c = MedicalLLMClient(_cfg(max_retries=3))
    with pytest.raises(ConfigurationError):
        c.ask_question("x")
    assert len(calls) == 1
    assert issubclass(ConfigurationError, ServerUnavailableError)


def test_openai_sdk_path_401_and_extra_body():
    import httpx
    import openai

    c = MedicalLLMClient({**_cfg(extra_body={"chat_template_kwargs": {"enable_thinking": False}}),
                          "server": {**_cfg()["server"], "client": "openai_sdk",
                                     "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}})
    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = types.SimpleNamespace(content="B")
            choice = types.SimpleNamespace(message=msg, finish_reason="stop")
            usage = types.SimpleNamespace(prompt_tokens=3, completion_tokens=1)
            return types.SimpleNamespace(choices=[choice], usage=usage, model="sdk-model")

    c._openai_client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))
    assert c.ask_question("x") == "B"
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert captured["seed"] == 42
    assert c.last_meta["model"] == "sdk-model" and c.last_meta["prompt_tokens"] == 3

    class _Denied:
        def create(self, **kwargs):
            resp = httpx.Response(401, request=httpx.Request("POST", "http://localhost:1/v1/chat/completions"))
            raise openai.AuthenticationError("bad key", response=resp, body=None)

    c._openai_client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Denied()))
    with pytest.raises(ConfigurationError):
        c.ask_question("x")


def test_programming_errors_are_not_swallowed(monkeypatch):
    def broken_post(*a, **k):
        raise NameError("bug")

    monkeypatch.setattr(requests, "post", broken_post)
    c = MedicalLLMClient(_cfg())
    with pytest.raises(NameError):
        c.ask_question("x")


def test_health_check(monkeypatch):
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: _FakeResponse(200, {"data": [{"id": "m"}, {"id": "other"}]}))
    assert MedicalLLMClient(_cfg()).health_check() == ["m", "other"]
    with pytest.raises(ConfigurationError, match="not served"):
        MedicalLLMClient(_cfg(model_name="missing")).health_check()

    def down(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "get", down)
    assert MedicalLLMClient(_cfg()).health_check() is None      # warn and continue


def test_judge_client_uses_judge_block(monkeypatch):
    sent = {}
    monkeypatch.setattr(requests, "post",
                        lambda url, headers, json, timeout, verify: sent.update(json) or
                        _FakeResponse(200, _ok_payload(content="1")))
    config = {
        "server": {"url": "http://a:1/v1", "model_name": "m", "seed": 5},
        "benchmark_settings": {"temperature": 0.7, "max_tokens": 64},
        "judge": {"url": "http://b:1/v1", "model_name": "judge", "client": "requests",
                  "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    }
    judge = main_mod._build_judge_client(config)
    assert (judge.temperature, judge.max_tokens, judge.seed) == (0, 512, 5)
    judge.ask_question("x")
    assert sent["model"] == "judge"
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    config["judge"].update(temperature=0.2, max_tokens=32)
    judge = main_mod._build_judge_client(config)
    assert (judge.temperature, judge.max_tokens) == (0.2, 32)


# ---------------------------------------------------------------------------
# Run dir: locks, dedupe, fingerprint
# ---------------------------------------------------------------------------

def test_benchmark_lock(tmp_path):
    a = run_utils.BenchmarkLocks(str(tmp_path), ["medqa", "rar"]).acquire()
    with pytest.raises(run_utils.LockHeldError, match="already running"):
        run_utils.BenchmarkLocks(str(tmp_path), ["radbench", "rar"]).acquire()
    # the failed attempt must not keep the lock it got before failing
    other = run_utils.BenchmarkLocks(str(tmp_path), ["radbench"]).acquire()
    other.release()
    a.release()
    run_utils.BenchmarkLocks(str(tmp_path), ["rar"]).acquire().release()


def test_main_fails_fast_when_locked(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"server": {"url": "http://localhost:1/v1", "model_name": "m"}}))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    held = run_utils.BenchmarkLocks(str(run_dir), ["medqa"]).acquire()
    try:
        rc = main_mod.main(["--config", str(cfg), "--run-dir", str(run_dir),
                            "--benchmark", "medqa", "--skip-health-check"])
    finally:
        held.release()
    assert rc == 2
    assert "already running" in capsys.readouterr().err
    assert not any(f.endswith(".log") for f in os.listdir(run_dir))


def test_dedupe_and_restrict_to_current_ids(tmp_path):
    path = tmp_path / "radbench_results.csv"
    pd.DataFrame({"id": ["1", "2", "1", "3", "9"],
                  "model_answer": ["old", "x", "new", "Error: boom", "y"]}).to_csv(path, index=False)
    eval_path, n_done, n_err, cleanup = main_mod._prepare_results_for_eval(
        str(path), {"1", "2", "3"}, str(tmp_path), judge_model="org/Judge-1")
    df = pd.read_csv(path, dtype=str)
    assert df["id"].tolist() == ["2", "1", "3", "9"]            # dedupe, last answer kept
    assert df.loc[df["id"] == "1", "model_answer"].item() == "new"
    assert (n_done, n_err) == (3, 1)
    assert eval_path != str(path)
    assert pd.read_csv(eval_path, dtype=str)["id"].tolist() == ["2", "1", "3"]
    from evaluate import _judge_cache_path
    cache_name = os.path.basename(_judge_cache_path(str(path), "org/Judge-1"))
    assert cache_name != "radbench_judge_cache.csv"
    link = os.path.join(os.path.dirname(eval_path), cache_name)
    assert os.path.realpath(link) == os.path.realpath(tmp_path / cache_name)
    cleanup()
    assert not os.path.exists(tmp_path / ".eval_subset")

    eval_path, n_done, _, cleanup = main_mod._prepare_results_for_eval(
        str(path), {"1", "2", "3", "9"}, str(tmp_path))
    assert eval_path == str(path) and n_done == 4


def test_drop_error_rows_atomic(tmp_path):
    path = tmp_path / "r.csv"
    pd.DataFrame({"id": ["1", "2"], "model_answer": ["A", "Error: x"]}).to_csv(path, index=False)
    main_mod._drop_error_rows(str(path))
    assert pd.read_csv(path, dtype=str)["id"].tolist() == ["1"]
    assert [f for f in os.listdir(tmp_path) if f.endswith(".tmp")] == []


def test_fingerprint_refuses_model_change(tmp_path, capsys):
    cfg = {"server": {"url": "https://h:4000/v1", "model_name": "a"}, "judge": {"model_name": "j"},
           "benchmark_settings": {"temperature": 0, "max_tokens": 256}}
    run_utils.check_fingerprint(str(tmp_path), cfg, ["medqa"])
    assert json.loads((tmp_path / "fingerprint.json").read_text())["model_name"] == "a"

    other = {**cfg, "server": {**cfg["server"], "model_name": "b"}}
    with pytest.raises(RuntimeError, match="model_name"):
        run_utils.check_fingerprint(str(tmp_path), other, ["medqa"])
    run_utils.check_fingerprint(str(tmp_path), other, ["medqa"], force=True)

    judge = {**cfg, "judge": {"model_name": "j2"}}
    with pytest.raises(RuntimeError, match="judge_model"):
        run_utils.check_fingerprint(str(tmp_path), judge, ["medqa"])

    soft = {**cfg, "benchmark_settings": {"temperature": 0.5, "max_tokens": 256}}
    capsys.readouterr()
    run_utils.check_fingerprint(str(tmp_path), soft, ["medqa", "rar"])
    assert "temperature" in capsys.readouterr().out
    assert "rar" in json.loads((tmp_path / "fingerprint.json").read_text())["max_tokens"]


def test_run_info_has_no_secrets():
    cfg = {"server": {"api_key": "sk-secret", "url": "u", "extra_body": {"token": "t"}},
           "judge": {"api_key": "sk-judge"}}
    info = run_utils.build_run_info(cfg, ["medqa"], config_path="config.yaml")
    text = json.dumps(info)
    assert "sk-secret" not in text and "sk-judge" not in text and '"t"' not in text
    assert info["hostname"] and info["python"] and "git_commit" in info
    assert "pandas" in info["packages"]


# ---------------------------------------------------------------------------
# End-to-end with a fake task: partial run is flagged
# ---------------------------------------------------------------------------

def test_partial_run_is_flagged(tmp_path, monkeypatch):
    data = [{"id": str(i), "benchmark": "MedQA", "question": f"q{i}", "correct_answer": "A",
             "options": [{"key": "A", "value": "a"}, {"key": "B", "value": "b"}]} for i in range(5)]

    def fake_loader(limit=None):
        return data[:limit] if limit else data

    def fake_run(config, client, items, results_path, logger=None):
        import csv
        with open(results_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "benchmark", "question", "correct_answer", "model_answer"])
            w.writeheader()
            for item in items[:3]:
                w.writerow({"id": item["id"], "benchmark": "MedQA", "question": item["question"],
                            "correct_answer": "A", "model_answer": "A"})
            w.writerow({"id": "0", "benchmark": "MedQA", "question": "q0",   # duplicate
                        "correct_answer": "A", "model_answer": "B"})
        raise ServerUnavailableError("10 consecutive request failures")

    monkeypatch.setitem(sys.modules, "fake_task_mod", types.SimpleNamespace(run=fake_run))
    monkeypatch.setattr(main_mod, "_registry",
                        lambda: {"medqa": (fake_loader, "fake_task_mod", "mcq"),
                                 "rar": (fake_loader, "fake_task_mod", "mcq")})
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"server": {"url": "http://localhost:1/v1", "model_name": "x",
                                              "api_key": "sk-secret"},
                                   "benchmark_settings": {"max_tokens": 8}}))
    run_dir = tmp_path / "run"
    rc = main_mod.main(["--config", str(cfg), "--model", "override-model", "--run-dir", str(run_dir),
                        "--benchmark", "medqa,rar", "--skip-health-check"])
    assert rc == 1
    status = json.loads((run_dir / "medqa_status.json").read_text())
    assert status["complete"] is False and status["n_items"] == 3 and status["n_expected"] == 5
    assert "ServerUnavailableError" in status["stop_reason"]
    assert not (run_dir / "rar_status.json").exists()              # skipped after the outage
    report = [json.loads(l) for l in (run_dir / "medqa_report.jsonl").read_text().splitlines()]
    acc = [r for r in report if r.get("metric") == "accuracy_pct"][0]["value"]
    assert acc == pytest.approx(100 * 2 / 3, abs=0.01)               # duplicate id 0 -> last answer "B"

    log = [f for f in os.listdir(run_dir) if f.endswith(".log")][0]
    log_text = (run_dir / log).read_text()
    assert "INCOMPLETE" in log_text and "skipped" in log_text
    info_files = [f for f in os.listdir(run_dir) if f.startswith("run_info_")]
    assert len(info_files) == 1
    info = json.loads((run_dir / info_files[0]).read_text())
    assert info["config"]["server"]["model_name"] == "override-model"
    assert info["config_path"].endswith("c.yaml")
    assert "sk-secret" not in json.dumps(info)
    assert info["status"]["medqa"]["complete"] is False
    assert json.loads((run_dir / "fingerprint.json").read_text())["model_name"] == "override-model"
    assert any(f.startswith("benchmark_results_") for f in os.listdir(run_dir))
    assert not any(f.endswith(".lock") and os.path.getsize(run_dir / f) == 0 for f in os.listdir(run_dir))


# ---------------------------------------------------------------------------
# Summary flattening, printout, plot
# ---------------------------------------------------------------------------

def test_flatten_carries_cis():
    rows = [
        {"type": "metric", "metric": "accuracy_pct", "value": 80.0, "ci_lo": 75.0, "ci_hi": 85.0,
         "ci_method": "wilson", "n": 100},
        {"type": "metric", "subset": "open", "metric": "llm_judge_accuracy_pct", "value": 50.0, "n_judged": 10},
        {"type": "metric", "subset": "open:category=plane", "metric": "accuracy_pct", "value": 40.0},
        {"type": "metric", "field": "birads_li", "metric": "accuracy_pct", "value": 70.0,
         "ci_lo": 60.0, "ci_hi": 80.0},
        {"type": "metric", "field": "lesions_li", "lesion_view": "type", "metric": "micro_f1_pct", "value": 61.0},
        {"type": "metric", "field": "lesions_li", "lesion_view": "count", "metric": "micro_f1_pct", "value": 55.0},
        {"type": "metric", "metric": "macro_f1_pct", "value": None},
        {"type": "region_metric", "region": "elbow", "micro_f1_pct": 90.0},
        {"type": "item", "id": "1", "is_correct": True},
    ]
    flat = flatten_report_rows(rows)
    assert flat["accuracy_pct_ci_lo"] == 75.0 and flat["accuracy_pct_ci_hi"] == 85.0
    assert flat["accuracy_pct_ci_method"] == "wilson" and flat["accuracy_pct_n"] == 100
    assert flat["open_judge_accuracy_pct"] == 50.0 and flat["open_judge_accuracy_pct_n"] == 10
    assert flat["open_category_plane_accuracy_pct"] == 40.0
    assert flat["birads_li_accuracy_pct_ci_lo"] == 60.0
    assert flat["lesions_li_micro_f1_pct"] == 61.0 and flat["lesions_li_count_micro_f1_pct"] == 55.0
    assert "macro_f1_pct" not in flat and flat["elbow_micro_f1_pct"] == 90.0

    merged = merge_metrics({"accuracy_pct": 80.5, "path": "x"}, flat)
    assert merged["accuracy_pct"] == 80.5 and merged["accuracy_pct_ci_hi"] == 85.0 and "path" not in merged
    line = format_metrics_line({"complete": False, "n_items": 3, "n_expected": 5, **merged},
                               headline_keys=["accuracy_pct"])
    assert line.startswith("INCOMPLETE (3/5") and "[75.0–85.0]" in line


def test_plot_with_incomplete_benchmark(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from core.plot import collect_bars, generate_results_chart

    summary = [
        ("medqa", {"accuracy_pct": 80.0, "accuracy_pct_ci_lo": 77.0, "accuracy_pct_ci_hi": 83.0,
                   "n_items": 100, "n_expected": 100, "complete": True}, None),
        ("radbench", {"mcq_accuracy_pct": 50.0, "mcq_rows": 63, "yes_no_accuracy_pct": 60.0,
                      "open_wbss_pct": 70.0, "n_items": 40, "n_expected": 285, "complete": False}, None),
        ("label_extraction_arm", {"micro_f1_pct": 80.0, "macro_f1_pct": 70.0, "accuracy_pct": 99.0,
                                  "complete": True}, None),
        ("rar", {}, "boom"),
    ]
    panels = collect_bars(summary)
    keys = [(b["bench"], b["key"]) for _, _, bars in panels for b in bars]
    assert ("radbench", "open_wbss_pct") not in keys                  # WBSS not a headline metric
    assert ("label_extraction_arm", "accuracy_pct") not in keys
    rb = [b for _, _, bars in panels for b in bars if b["bench"] == "radbench"]
    assert all(b["incomplete"] for b in rb) and rb[0]["n"] == 63 and rb[0]["chance"] is None
    assert [b for _, _, bars in panels for b in bars if b["bench"] == "medqa"][0]["chance"] == 25.0

    out = tmp_path / "chart.png"
    generate_results_chart(summary, "m", str(out))
    assert out.exists() and out.stat().st_size > 10_000


# ---------------------------------------------------------------------------
# Logger, compare script
# ---------------------------------------------------------------------------

def test_logger_tees_stderr(tmp_path):
    log = tmp_path / "run.log"
    with RunLogger(str(log)) as logger:
        print("to stdout")
        print("to stderr", file=sys.stderr)
        logger.verbose("only file")
        assert sys.stdout.isatty() in (True, False)
        assert sys.stdout.encoding
    with RunLogger(str(log)):
        print("second session")
    text = log.read_text()
    assert "to stdout" in text and "to stderr" in text and "only file" in text
    assert "second session" in text and text.count("=== Run Log") == 2   # append mode


def test_compare_models(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import compare_models as cm

    def make_run(name, model, correct):
        d = tmp_path / name
        d.mkdir()
        (d / "run_info_x.json").write_text(json.dumps({"config": {"server": {"model_name": model}}}))
        lines = [{"type": "metric", "metric": "accuracy_pct", "value": 100 * sum(correct) / len(correct)}]
        lines += [{"type": "item", "id": str(i), "is_correct": bool(c)} for i, c in enumerate(correct)]
        (d / "medqa_report.jsonl").write_text("\n".join(json.dumps(l) for l in lines))
        (d / "label_extraction_arm_report.jsonl").write_text(
            json.dumps({"type": "metric", "metric": "verbatim_citation_rate_pct", "value": 90.0}) + "\n" +
            json.dumps({"type": "metric", "metric": "micro_f1_pct", "value": 80.0, "ci_lo": 78.0, "ci_hi": 82.0}))
        return str(d)

    a = make_run("a", "model-a", [1] * 20)
    b = make_run("b", "model-b", [0] * 10 + [1] * 10)
    out = tmp_path / "out"
    assert cm.main([a, b, "--out-dir", str(out)]) == 0
    comp = pd.read_csv(out / "comparison.csv")
    cite = comp[comp["metric"] == "verbatim_citation_rate_pct"]
    assert sorted(cite["model"]) == ["model-a", "model-b"]
    f1 = comp[(comp["metric"] == "micro_f1_pct") & (comp["model"] == "model-a")].iloc[0]
    assert (f1["ci_lo"], f1["ci_hi"]) == (78.0, 82.0)
    mc = pd.read_csv(out / "mcnemar.csv").iloc[0]
    assert (mc["only_a_correct"], mc["only_b_correct"], mc["n_shared"]) == (10, 0, 20)
    assert mc["p_value"] == pytest.approx(cm.mcnemar_exact(10, 0), abs=1e-6) and mc["p_value"] < 0.01
    assert (out / "comparison.png").exists()
    assert cm.mcnemar_exact(0, 0) == 1.0 and cm.mcnemar_exact(5, 5) == 1.0


def test_flat_key_ignores_judge_model_and_variant():
    from core.summary import flatten_report_rows
    rows = [
        {"type": "metric", "subset": "open", "metric": "llm_judge_accuracy_pct", "value": 50.0,
         "ci_lo": 40.0, "ci_hi": 60.0, "judge_model": "org/Some-Judge"},
        {"type": "metric", "field": "birads_li", "metric": "accuracy_birads6keep_pct",
         "value": 34.4, "variant": "birads6_keep"},
    ]
    flat = flatten_report_rows(rows)
    assert flat["open_judge_accuracy_pct"] == 50.0
    assert flat["open_judge_accuracy_pct_ci_lo"] == 40.0
    assert flat["birads_li_accuracy_birads6keep_pct"] == 34.4


def test_limit_must_be_positive(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("server: {url: 'https://x/v1', model_name: m}\nbenchmark: medqa\n")
    args = main_mod._parse_args(["--config", str(cfg), "--limit", "0"])
    with pytest.raises(SystemExit):
        main_mod._load_config(args)


def test_fingerprint_tracks_input_code():
    from core.run_utils import build_fingerprint
    fp = build_fingerprint({"server": {"model_name": "m"}}, ["medqa"])
    assert len(fp["input_code_sha256"]) == 16
