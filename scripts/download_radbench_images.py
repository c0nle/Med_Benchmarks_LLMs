"""
Download the Radiopaedia images referenced by data/radbench.csv into the RadBench
image cache (data/radbench_images_v2/) used by loaders/vision_benchmarks.py.

Each image is stored as sha1(reference)[:16] + original extension, so different URLs
never share a file (the last URL segment is not unique: 4 URLs end in
"0._jumbo.jpeg"). A manifest CSV (reference, kind, file, sha256, bytes, status) is
written next to the images.

Only http(s) references are downloaded. MedPix UUIDs (MedPix is currently offline)
and bare ids such as "52662257" (case 77654) are listed in the manifest as
"not_downloadable"; the loader drops those questions and reports them.

Usage:
    python scripts/download_radbench_images.py            # download missing images
    python scripts/download_radbench_images.py --force    # re-download all
    python scripts/download_radbench_images.py --check    # only verify cache + manifest

Exit code 1 if a downloadable image is missing after the run.
"""
import argparse
import csv
import hashlib
import io
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from loaders.vision_benchmarks import (  # noqa: E402
    _RADBENCH_IMAGE_DIR,
    _RADBENCH_PATH,
    radbench_image_kind,
    radbench_image_path,
    radbench_image_refs,
)

USER_AGENT = ("Med_Benchmarks_LLMs/1.0 (academic benchmark; fetches the RadBench "
              "evaluation images once; python-requests)")
MANIFEST = _RADBENCH_IMAGE_DIR / "manifest.csv"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_image(data: bytes) -> bool:
    from PIL import Image
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        return True
    except Exception:
        return False


def _download(session, url: str, retries: int, backoff_s: float, timeout_s: float) -> bytes:
    last = None
    for attempt in range(retries + 1):
        try:
            r = session.get(url, timeout=timeout_s)
            if r.status_code == 200 and r.content and _is_image(r.content):
                return r.content
            last = f"HTTP {r.status_code}, {r.headers.get('content-type')}, {len(r.content)} bytes"
            if r.status_code in (403, 404, 410):
                break                      # permanent, do not hammer the server
        except Exception as e:             # timeouts, connection errors
            last = str(e)
        if attempt < retries:
            time.sleep(backoff_s * 2 ** attempt)
    raise RuntimeError(last or "unknown error")


def collect_references(csv_path=_RADBENCH_PATH) -> dict:
    """reference → sorted list of question ids (CASE_ID-q<row>) that use it."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    refs = defaultdict(list)
    for row_no, row in df.iterrows():
        for ref in radbench_image_refs(row.get("imageIDs")):
            refs[ref].append(f"{row.get('CASE_ID') or 'radbench'}-q{row_no}")
    return dict(refs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download images that are already cached")
    ap.add_argument("--check", action="store_true", help="do not download, only verify and rewrite the manifest")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests (default 1)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=60)
    args = ap.parse_args()

    refs = collect_references()
    os.makedirs(_RADBENCH_IMAGE_DIR, exist_ok=True)

    # File names must be unique per reference
    by_file = defaultdict(list)
    for ref in refs:
        by_file[radbench_image_path(ref).name].append(ref)
    clashes = {f: r for f, r in by_file.items() if len(r) > 1}
    if clashes:
        raise SystemExit(f"File name collision in cache naming: {clashes}")

    session = None
    if not args.check:
        import requests
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT

    rows, failed = [], []
    urls = [r for r in refs if radbench_image_kind(r) == "url"]
    print(f"{len(refs)} unique image references in {_RADBENCH_PATH}: {len(urls)} URLs, "
          f"{sum(radbench_image_kind(r) == 'medpix' for r in refs)} MedPix ids, "
          f"{sum(radbench_image_kind(r) == 'unresolvable' for r in refs)} other")

    n_new = 0
    for ref in sorted(refs):
        kind = radbench_image_kind(ref)
        path = radbench_image_path(ref)
        status = "cached" if path.exists() else "missing"
        if kind == "url" and session is not None and (args.force or not path.exists()):
            try:
                data = _download(session, ref, args.retries, 5.0, args.timeout)
                tmp = path.with_suffix(path.suffix + ".part")
                tmp.write_bytes(data)
                os.replace(tmp, path)
                status = "downloaded"
                n_new += 1
            except Exception as e:
                status = f"failed: {e}"
                print(f"  FAILED {ref}: {e}")
            time.sleep(args.delay)
        elif kind != "url" and not path.exists():
            status = "not_downloadable"

        data = path.read_bytes() if path.exists() else b""
        if kind == "url" and not data:
            failed.append(ref)
        rows.append({
            "reference": ref,
            "kind": kind,
            "file": path.name if data else "",
            "sha256": _sha256(data) if data else "",
            "bytes": len(data),
            "status": status,
            "questions": " ".join(refs[ref]),
        })

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Identical content under different references (reported, not an error)
    by_hash = defaultdict(list)
    for r in rows:
        if r["sha256"]:
            by_hash[r["sha256"]].append(r["reference"])
    dupes = {h: r for h, r in by_hash.items() if len(r) > 1}

    n_cached = sum(1 for r in rows if r["kind"] == "url" and r["sha256"])
    print(f"Downloaded {n_new}; {n_cached}/{len(urls)} URL images present in {_RADBENCH_IMAGE_DIR}/")
    print(f"Manifest: {MANIFEST}")
    for h, rr in dupes.items():
        print(f"  NOTE identical image bytes for different references: {rr}")
    not_dl = [r for r in rows if r["status"] == "not_downloadable"]
    if not_dl:
        print(f"  {len(not_dl)} references are not downloadable (MedPix / bare ids); "
              "their questions are dropped by the loader")
    if failed:
        print(f"ERROR: {len(failed)} URL image(s) missing: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
