"""
Download and parse the PHEME rumour/fake-news dataset from figshare.

Labels:
  1 = real  (non-rumour threads OR rumours verified as TRUE)
  0 = fake  (rumours verified as FALSE)
  unverified rumours are skipped — too noisy for binary classification.

Output: data/raw/pheme_tweets.csv  (columns: text, label, event, thread_id)
"""

import io
import json
import os
import tarfile
import time
import zipfile

import pandas as pd
import requests

FIGSHARE_ARTICLE_ID = 6392078
RAW_OUT = "data/raw/pheme_tweets.csv"
EXTRACTED_DIR = "data/raw/all-rnr-annotated-threads"


def _normalise_tweet(text: str) -> str:
    import re
    text = re.sub(r"http\S+|www\S+", "HTTPURL", text)
    text = re.sub(r"@\w+", "@USER", text)
    return text.strip()


# ── Parse from already-extracted directory ─────────────────────────────────────

def _parse_dir(root: str) -> pd.DataFrame:
    """Parse PHEME from the extracted all-rnr-annotated-threads directory."""
    records = []
    for event_dir in os.listdir(root):
        event_path = os.path.join(root, event_dir)
        if not os.path.isdir(event_path):
            continue
        event = event_dir.split("-all-rnr")[0].split("-")[0]

        for rumour_type in ("rumours", "non-rumours"):
            type_path = os.path.join(event_path, rumour_type)
            if not os.path.isdir(type_path):
                continue

            for thread_id in os.listdir(type_path):
                thread_path = os.path.join(type_path, thread_id)

                if rumour_type == "non-rumours":
                    label = 1
                else:
                    ann_path = os.path.join(thread_path, "annotation.json")
                    if not os.path.exists(ann_path):
                        continue
                    try:
                        with open(ann_path) as f:
                            ann = json.load(f)
                        veracity = ann.get("veracity", {}).get("value", "unverified")
                    except Exception:
                        continue
                    if veracity == "false":
                        label = 0
                    elif veracity == "true":
                        label = 1
                    else:
                        continue  # skip unverified

                src_dir = os.path.join(thread_path, "source-tweets")
                if not os.path.isdir(src_dir):
                    continue
                for fname in os.listdir(src_dir):
                    if not fname.endswith(".json"):
                        continue
                    try:
                        with open(os.path.join(src_dir, fname)) as f:
                            tweet = json.load(f)
                        text = tweet.get("text", "").strip()
                        if not text:
                            continue
                    except Exception:
                        continue
                    records.append({
                        "text": _normalise_tweet(text),
                        "label": label,
                        "event": event,
                        "thread_id": thread_id,
                    })

    return pd.DataFrame(records)


# ── Parse from in-memory archive (zip or tar.gz) ───────────────────────────────

def _parse_archive(data: bytes) -> pd.DataFrame:
    """Try zip first, then tar.gz."""
    # Try zip
    try:
        return _parse_zip(data)
    except zipfile.BadZipFile:
        pass
    # Try tar.gz
    try:
        return _parse_tar(data)
    except Exception:
        pass
    raise RuntimeError("Downloaded file is neither a zip nor a tar.gz")


def _parse_zip(zip_bytes: bytes) -> pd.DataFrame:
    records = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        source_paths = [n for n in names if "source-tweets" in n and n.endswith(".json")]
        for src_path in source_paths:
            parts = src_path.replace("\\", "/").split("/")
            event = parts[0].split("-all-rnr")[0].split("-")[0] if parts else "unknown"
            if "non-rumours" in src_path:
                label = 1
                thread_id = parts[parts.index("non-rumours") + 1]
            elif "rumours" in src_path:
                thread_id = parts[parts.index("rumours") + 1]
                base = src_path.split("/source-tweets/")[0]
                ann_path = base + "/annotation.json"
                if ann_path not in names:
                    continue
                try:
                    ann = json.loads(zf.read(ann_path))
                    veracity = ann.get("veracity", {}).get("value", "unverified")
                except Exception:
                    continue
                if veracity == "false":
                    label = 0
                elif veracity == "true":
                    label = 1
                else:
                    continue
            else:
                continue
            try:
                tweet = json.loads(zf.read(src_path))
                text = tweet.get("text", "").strip()
                if not text:
                    continue
            except Exception:
                continue
            records.append({"text": _normalise_tweet(text), "label": label,
                            "event": event, "thread_id": thread_id})
    return pd.DataFrame(records)


def _parse_tar(tar_bytes: bytes) -> pd.DataFrame:
    records = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
        names = tf.getnames()
        source_paths = [n for n in names if "source-tweets" in n and n.endswith(".json")]
        for src_path in source_paths:
            parts = src_path.replace("\\", "/").split("/")
            event = parts[0].split("-all-rnr")[0].split("-")[0] if parts else "unknown"
            if "non-rumours" in src_path:
                label = 1
                thread_id = parts[parts.index("non-rumours") + 1]
            elif "rumours" in src_path:
                thread_id = parts[parts.index("rumours") + 1]
                base = src_path.split("/source-tweets/")[0]
                ann_path = base + "/annotation.json"
                if ann_path not in names:
                    continue
                try:
                    ann = json.loads(tf.extractfile(tf.getmember(ann_path)).read())
                    veracity = ann.get("veracity", {}).get("value", "unverified")
                except Exception:
                    continue
                if veracity == "false":
                    label = 0
                elif veracity == "true":
                    label = 1
                else:
                    continue
            else:
                continue
            try:
                tweet = json.loads(tf.extractfile(tf.getmember(src_path)).read())
                text = tweet.get("text", "").strip()
                if not text:
                    continue
            except Exception:
                continue
            records.append({"text": _normalise_tweet(text), "label": label,
                            "event": event, "thread_id": thread_id})
    return pd.DataFrame(records)


# ── Download ───────────────────────────────────────────────────────────────────

def _get_download_url() -> str:
    api_url = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}/files"
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    files = resp.json()
    for f in files:
        if f["name"].endswith(".zip"):
            return f["download_url"]
    return files[0]["download_url"]


def download_pheme() -> pd.DataFrame:
    # Fast path: already extracted on disk
    if os.path.isdir(EXTRACTED_DIR):
        print(f"Found extracted PHEME directory — parsing from disk …")
        return _parse_dir(EXTRACTED_DIR)

    print("Fetching PHEME download URL from figshare API …")
    url = _get_download_url()
    print(f"Downloading from: {url}")

    for attempt in range(10):
        resp = requests.get(url, stream=True, timeout=180)
        if resp.status_code == 202:
            print(f"  figshare preparing file (202), retrying in 5s [{attempt+1}/10]")
            time.sleep(5)
            continue
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        chunks, downloaded = [], 0
        for chunk in resp.iter_content(chunk_size=1 << 20):
            chunks.append(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded/1e6:.1f} / {total/1e6:.1f} MB", end="", flush=True)
        print()
        data = b"".join(chunks)
        print(f"Downloaded {len(data)/1e6:.1f} MB — parsing …")
        return _parse_archive(data)

    raise RuntimeError("figshare did not serve the file after 10 attempts")


def main():
    os.makedirs("data/raw", exist_ok=True)
    df = download_pheme()
    print(f"\nTotal records : {len(df)}")
    print(f"  Fake  (0)   : {(df['label'] == 0).sum()}")
    print(f"  Real  (1)   : {(df['label'] == 1).sum()}")
    print(f"  Events      : {df['event'].unique().tolist()}")
    df.to_csv(RAW_OUT, index=False)
    print(f"\nSaved to {RAW_OUT}")


if __name__ == "__main__":
    main()
