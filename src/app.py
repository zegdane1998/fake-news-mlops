import os
import pickle
from datetime import datetime

import pandas as pd
import torch
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128
MASTER_CSV = "data/new_scraped/all_tweets.csv"

app = FastAPI()
templates = Jinja2Templates(directory="templates")
templates.env.globals["zip"] = zip

_tokenizer = AutoTokenizer.from_pretrained("models/bertweet_finetuned", use_fast=False)
_model = AutoModelForSequenceClassification.from_pretrained("models/bertweet_finetuned").to(DEVICE)
_model.eval()


def _predict_batch(texts: list[str]) -> list[float]:
    enc = _tokenizer(
        texts,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = _model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
        ).logits
    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
    return probs


def _get_data_file():
    if os.path.exists(MASTER_CSV):
        return MASTER_CSV
    scrape_dir = "data/new_scraped"
    if not os.path.exists(scrape_dir):
        return None
    files = [os.path.join(scrape_dir, f) for f in os.listdir(scrape_dir) if f.endswith(".csv")]
    return max(files, key=os.path.getmtime) if files else None


def get_pipeline_status():
    data_file = _get_data_file()
    if data_file is None:
        return {"last_sync": "No data yet", "status": "Waiting for scrape",
                "counts": [0, 0], "keywords": [], "keyword_counts": []}

    last_sync = datetime.fromtimestamp(os.path.getmtime(data_file)).strftime("%Y-%m-%d %H:%M")
    df = pd.read_csv(data_file).dropna(subset=["text"])
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
        "iran", "airstrike", "nuclear", "sanctions", "tehran",
        "houthis", "pentagon", "trump", "congress", "missile",
        "irgc", "persian gulf", "proxy war", "white house", "biden",
    ]
    keyword_counts = sorted(
        [(kw, all_text.count(kw)) for kw in tracked_keywords],
        key=lambda x: -x[1]
    )
    keyword_counts = [(kw, cnt) for kw, cnt in keyword_counts if cnt > 0][:8]

    return {
        "last_sync": last_sync,
        "status": "Healthy",
        "counts": [real_count, fake_count],
        "keywords": [k for k, _ in keyword_counts],
        "keyword_counts": [c for _, c in keyword_counts],
    }


def get_latest_tweets(n: int = 10):
    data_file = _get_data_file()
    if not data_file:
        return []
    try:
        df = pd.read_csv(data_file).dropna(subset=["text"])
        if "scraped_at" in df.columns:
            df = df.sort_values("scraped_at", ascending=False)
        df = df.head(n)
        texts = df["text"].astype(str).tolist()
        probs = _predict_batch(texts)

        tweets = []
        for (_, row), prob in zip(df.iterrows(), probs):
            conf = prob if prob > 0.5 else 1 - prob
            tweets.append({
                "text": row["text"],
                "scraped_at": row.get("scraped_at", "N/A"),
                "source": row.get("source", "NewsAPI"),
                "verdict": "Real" if prob > 0.5 else "Fake",
                "conf": f"{conf * 100:.1f}%",
            })
        return tweets
    except Exception as e:
        print(f"Error loading tweets: {e}")
        return []


@app.get("/")
async def home(request: Request):
    pipeline = get_pipeline_status()
    tweets = get_latest_tweets()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "last_sync": pipeline["last_sync"],
        "status": pipeline["status"],
        "region": "United States",
        "stats": {
            "labels": ["Real", "Fake"],
            "counts": pipeline["counts"],
            "keywords": pipeline["keywords"],
            "keyword_counts": pipeline["keyword_counts"],
        },
        "tweets": tweets,
        "result": None,
        "headline": None,
    })


@app.post("/analyze")
async def analyze(request: Request, headline: str = Form(...)):
    prob = _predict_batch([headline])[0]
    conf = prob if prob > 0.5 else 1 - prob
    result = {"verdict": "Real" if prob > 0.5 else "Fake", "conf": f"{conf * 100:.1f}%"}

    pipeline = get_pipeline_status()
    tweets = get_latest_tweets()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "headline": headline,
        "last_sync": pipeline["last_sync"],
        "status": pipeline["status"],
        "region": "United States",
        "stats": {
            "labels": ["Real", "Fake"],
            "counts": pipeline["counts"],
            "keywords": pipeline["keywords"],
            "keyword_counts": pipeline["keyword_counts"],
        },
        "tweets": tweets,
    })
