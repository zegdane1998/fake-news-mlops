import os
import pickle
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 100  # must match src/train.py MAX_LEN

app = FastAPI()
templates = Jinja2Templates(directory="templates")
templates.env.globals["zip"] = zip  # expose zip() to Jinja2

MODEL = load_model("models/cnn_v1.h5")
with open("models/tokenizer.pkl", "rb") as handle:
    TOKENIZER = pickle.load(handle)

SCRAPE_DIR = "data/new_scraped"


def _get_latest_file():
    if not os.path.exists(SCRAPE_DIR):
        return None
    files = [
        os.path.join(SCRAPE_DIR, f)
        for f in os.listdir(SCRAPE_DIR)
        if f.endswith(".csv")
    ]
    return max(files, key=os.path.getmtime) if files else None


def _predict_batch(texts):
    sequences = TOKENIZER.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    return MODEL.predict(padded, verbose=0).flatten().tolist()


def get_pipeline_status():
    latest_file = _get_latest_file()
    if latest_file is None:
        return {
            "last_sync": "No data yet",
            "status": "Waiting for scrape",
            "counts": [0, 0],
            "keywords": [],
            "keyword_counts": [],
        }

    last_sync = datetime.fromtimestamp(
        os.path.getmtime(latest_file)
    ).strftime("%Y-%m-%d %H:%M")

    df = pd.read_csv(latest_file).dropna(subset=["text"])
    texts = df["text"].astype(str).tolist()

    if not texts:
        return {"last_sync": last_sync, "status": "Empty file",
                "counts": [0, 0], "keywords": [], "keyword_counts": []}

    probs = _predict_batch(texts)
    predictions = [1 if p > 0.5 else 0 for p in probs]
    real_count = sum(predictions)
    fake_count = len(predictions) - real_count

    all_text = " ".join(texts).lower()
    tracked_keywords = [
        "congress", "white house", "trump", "election", "senate",
        "ballots", "breaking", "leaked", "biden", "republican"
    ]
    keyword_counts = [(kw, all_text.count(kw)) for kw in tracked_keywords]
    keyword_counts = sorted(keyword_counts, key=lambda x: -x[1])
    keyword_counts = [(kw, cnt) for kw, cnt in keyword_counts if cnt > 0][:8]
    keywords = [k for k, _ in keyword_counts]
    kw_counts = [c for _, c in keyword_counts]

    return {
        "last_sync": last_sync,
        "status": "Healthy",
        "counts": [real_count, fake_count],
        "keywords": keywords,
        "keyword_counts": kw_counts,
    }


def get_latest_tweets(n=10):
    latest_file = _get_latest_file()
    if not latest_file:
        return []
    try:
        df = pd.read_csv(latest_file).dropna(subset=["text"]).tail(n)
        texts = df["text"].astype(str).tolist()
        probs = _predict_batch(texts)

        tweets = []
        for (_, row), pred in zip(df.iterrows(), probs):
            conf_val = pred if pred > 0.5 else 1 - pred
            tweets.append({
                "text": row["text"],
                "scraped_at": row.get("scraped_at", "N/A"),
                "source": row.get("source", "NewsAPI"),
                "verdict": "Real" if pred > 0.5 else "Fake",
                "conf": f"{conf_val * 100:.1f}%",
            })
        return tweets
    except Exception as e:
        print(f"Tweet loading error: {e}")
        return []


@app.get("/")
async def home(request: Request):
    pipeline = get_pipeline_status()
    tweets = get_latest_tweets(n=10)
    stats = {
        "labels": ["Real", "Fake"],
        "counts": pipeline["counts"],
        "keywords": pipeline["keywords"],
        "keyword_counts": pipeline["keyword_counts"],
    }
    return templates.TemplateResponse("index.html", {
        "request": request,
        "last_sync": pipeline["last_sync"],
        "status": pipeline["status"],
        "region": "United States",
        "stats": stats,
        "tweets": tweets,
        "result": None,
        "headline": None,
    })


@app.post("/analyze")
async def analyze(request: Request, headline: str = Form(...)):
    probs = _predict_batch([headline])
    pred = probs[0]
    conf_val = pred if pred > 0.5 else 1 - pred
    result = {
        "verdict": "Real" if pred > 0.5 else "Fake",
        "conf": f"{conf_val * 100:.1f}%",
    }
    pipeline = get_pipeline_status()
    tweets = get_latest_tweets(n=10)
    stats = {
        "labels": ["Real", "Fake"],
        "counts": pipeline["counts"],
        "keywords": pipeline["keywords"],
        "keyword_counts": pipeline["keyword_counts"],
    }
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "headline": headline,
        "last_sync": pipeline["last_sync"],
        "status": pipeline["status"],
        "region": "United States",
        "stats": stats,
        "tweets": tweets,
    })