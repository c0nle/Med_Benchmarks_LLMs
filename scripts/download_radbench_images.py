"""
Download all RadBench images (Radiopaedia + MedPix) to data/radbench_images/.

Usage:
    python3 scripts/download_radbench_images.py

Images are saved as:
    data/radbench_images/<safe_filename>.jpg

For Radiopaedia: the URL slug (last path component) is used as filename.
For MedPix: the UUID is used; the real URL is fetched first from the MedPix API.
"""
import os
import sys
import time
import urllib.request
import urllib.error
import json
from pathlib import Path

CSV_PATH = "data/radbench_repo/data/radbench/radbench.csv"
OUT_DIR  = Path("data/radbench_images")

MEDPIX_API = "https://medpix.nlm.nih.gov/rest/image.json?imageID={uuid}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def _download(url: str, dest: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                dest.write_bytes(resp.read())
            return True
        except urllib.error.HTTPError as e:
            print(f"    HTTP {e.code} — {url}")
            return False
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    FAILED after {retries} tries: {e}")
                return False
    return False


def _medpix_image_url(uuid: str):
    api_url = MEDPIX_API.format(uuid=uuid)
    try:
        req = urllib.request.Request(api_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return data.get("imageURL") or data.get("url") or None
    except Exception as e:
        print(f"    MedPix API error for {uuid}: {e}")
        return None


def _safe_name(image_id: str) -> str:
    """Derive a filesystem-safe filename from a URL or UUID."""
    name = image_id.split("/")[-1]          # last URL component, or UUID itself
    name = name.split("?")[0]               # strip query params
    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
        name += ".jpg"
    return name


def main():
    import pandas as pd

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # Collect unique (source, image_id) pairs
    seen = set()
    tasks = []
    for _, row in df.iterrows():
        source = str(row["imageSource"]).strip().lower()
        for img_id in str(row["imageIDs"]).split(","):
            img_id = img_id.strip()
            key = (source, img_id)
            if key not in seen:
                seen.add(key)
                tasks.append((source, img_id))

    print(f"{len(tasks)} unique images to download → {OUT_DIR}/")
    ok = fail = skip = 0

    for i, (source, img_id) in enumerate(tasks, 1):
        fname = _safe_name(img_id)
        dest  = OUT_DIR / fname

        if dest.exists():
            print(f"  [{i}/{len(tasks)}] SKIP (exists)  {fname}")
            skip += 1
            continue

        if source == "radiopaedia":
            url = img_id  # already a direct URL
        else:
            # MedPix: resolve UUID → image URL via API
            print(f"  [{i}/{len(tasks)}] MedPix API lookup: {img_id[:8]}…")
            url = _medpix_image_url(img_id)
            if not url:
                fail += 1
                continue

        print(f"  [{i}/{len(tasks)}] Downloading {fname} …", end=" ", flush=True)
        if _download(url, dest):
            print("OK")
            ok += 1
        else:
            fail += 1
        time.sleep(0.2)  # be polite

    print(f"\nDone. OK={ok}  SKIPPED={skip}  FAILED={fail}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
