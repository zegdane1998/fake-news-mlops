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
import zipfile

import pandas as pd
import requests

FIGSHARE_URL = (
    "https://figshare.com/ndownloader/articles/6392078/versions/1"
)
RAW_OUT = "data/raw/pheme_tweets.csv"


def _normalise_tweet(text: str) -> str:
    """Light normalisation that keeps BERTweet-friendly tokens."""
    import re
    text = re.sub(r"http\S+|www\S+", "HTTPURL", text)
    text = re.sub(r"@\w+", "@USER", text)
    return text.strip()


def download_pheme() -> pd.DataFrame:
    print("Downloading PHEME dataset from figshare …")
    resp = requests.get(FIGSHARE_URL, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    chunks = []
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB
        chunks.append(chunk)
        downloaded += len(chunk)
        if total:
            print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()

    zip_bytes = b"".join(chunks)
    print(f"Downloaded {len(zip_bytes) / 1e6:.1f} MB — parsing …")
    return _parse_zip(zip_bytes)


def _parse_zip(zip_bytes: bytes) -> pd.DataFrame:
    records = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()

        # Collect all source-tweet JSON paths
        source_paths = [n for n in names if "source-tweets" in n and n.endswith(".json")]

        for src_path in source_paths:
            parts = src_path.replace("\\", "/").split("/")

            # Determine event name (top-level folder)
            event = parts[0].split("-all-rnr")[0].split("-")[0] if parts else "unknown"

            # Determine if this thread is a rumour or non-rumour
            if "non-rumours" in src_path:
                label = 1          # definitively real
                thread_id = parts[parts.index("non-rumours") + 1]
            elif "rumours" in src_path:
                # Need the annotation to get veracity
                thread_id = parts[parts.index("rumours") + 1]
                ann_path = "/".join(src_path.split("/source-tweets")[0].split("/")[:-0])
                # Build annotation path
                base = src_path.split("/source-tweets/")[0]
                ann_path = base + "/annotation.json"
                if ann_path not in zf.namelist():
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
                    continue   # skip unverified
            else:
                continue

            # Read source tweet text
            try:
                tweet = json.loads(zf.read(src_path))
                text = tweet.get("text", "")
                if not text.strip():
                    continue
            except Exception:
                continue

            records.append({
                "text": _normalise_tweet(text),
                "label": label,
                "event": event,
                "thread_id": thread_id,
            })

    df = pd.DataFrame(records)
    return df


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
