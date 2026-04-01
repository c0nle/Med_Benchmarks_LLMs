## Med_Benchmarks_LLMs

Kleines Benchmark-Skript für Multiple-Choice-Textbenchmarks (aktuell MedQA) gegen einen OpenAI-kompatiblen Server (z.B. LiteLLM).

## Quickstart

```bash
pip install -r requirements.txt
cp config.default.yaml config.yaml
export MED_SERVER_API_KEY="dein_key"
python main.py
```

Ergebnisse:
- `results/benchmark_results.csv` (Rohdaten)
- `results/benchmark_report.jsonl` (komplette Auswertung in 1 Datei)

## MedQA Datenquelle

Standardmäßig lädt `loaders/text_benchmarks.py` den Benchmark von Hugging Face (`openlifescienceai/medqa`, Split `test`).
Wenn deine Umgebung kein Internet/DNS hat, lade `test-00000-of-00001.parquet` außerhalb herunter und setze:

```bash
export MEDQA_PARQUET_PATH=/pfad/zur/medqa-test.parquet
python main.py
```

## API-Key / Secrets

Lege API-Keys nicht in Git ab. Das Repo liest standardmäßig aus `MED_SERVER_API_KEY` (konfigurierbar über `server.api_key_env`).
`config.yaml` ist via `.gitignore` ausgeschlossen.

## SSL / Self-signed Zertifikate

Wenn dein Server HTTPS mit self-signed Zertifikat nutzt und du `CERTIFICATE_VERIFY_FAILED` siehst:
- `server.verify_ssl: false` (unsicher, aber ok für internes Testing) oder
- `server.verify_ssl: "certs/server-ca.pem"` (besser: CA/Cert-Datei angeben).

## Auswertung / Score (CLI)

```bash
python evaluate.py results/benchmark_results.csv --out results/benchmark_report.jsonl
```

## Full Run (ganzer Datensatz)

Setze in `config.yaml`:
- `benchmark_settings.limit_samples: null`
- optional: `benchmark_settings.sleep_s: 0.1` (wenn du Ratelimits/Überlast siehst)

Und starte dann `python main.py`. Der Lauf kann je nach Datensatzgröße/Server-Ratelimit Stunden dauern; das Script schreibt die CSV **inkrementell** und setzt automatisch fort, wenn `results/benchmark_results.csv` bereits existiert.

## Clean (pycache)

```bash
python scripts/clean.py
```
