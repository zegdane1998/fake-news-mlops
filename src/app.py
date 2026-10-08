import json
import os
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

_tokenizer = AutoTokenizer.from_pretrained("models/bertweet_finetuned", use_fast=True)
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

    try:
        sample = texts[:50]
        probs = _predict_batch(sample)
        predictions = [1 if p > 0.5 else 0 for p in probs]
        real_count = sum(predictions)
        fake_count = len(predictions) - real_count
    except Exception:
        real_count, fake_count = 0, 0

    all_text = " ".join(texts).lower()
    tracked_keywords = [
        # Iran / Middle East
        "iran", "tehran", "irgc", "khamenei", "persian gulf",
        "hormuz", "strait", "tanker", "warship", "naval",
        "houthis", "hezbollah", "hamas", "proxy war", "airstrike",
        # Nuclear
        "nuclear", "uranium", "enrichment", "jcpoa", "missile",
        # US Politics
        "trump", "biden", "harris", "congress", "senate",
        "pentagon", "white house", "sanctions", "tariff", "ceasefire",
        # Broader conflict
        "israel", "gaza", "netanyahu", "oil", "drone",
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
        try:
            probs = _predict_batch(texts)
        except Exception:
            probs = [0.5] * len(texts)

        tweets = []
        for (_, row), prob in zip(df.iterrows(), probs):
            conf = prob if prob > 0.5 else 1 - prob
            tweets.append({
                "text": row["text"],
                "scraped_at": row.get("scraped_at", "N/A"),
                "source": row.get("source", "NewsAPI"),
                "verdict": "Real" if prob > 0.5 else "Fake",
                "conf": f"{conf * 100:.1f}%",
                "conf_num": round(conf * 100, 1),
            })
        return tweets
    except Exception as e:
        print(f"Error loading tweets: {e}")
        return []


def get_model_metrics():
    path = "metrics/retraining_comparison.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        m = d.get("pheme_only", {})
        return {
            "accuracy": round(m.get("accuracy", 0) * 100, 2),
            "f1_fake":  round(m.get("f1_fake", 0) * 100, 2),
            "f1_real":  round(m.get("f1_real", 0) * 100, 2),
            "auc_roc":  round(m.get("auc_roc", 0) * 100, 2),
            "n_test":   d.get("n_test", 0),
            "n_pseudo": d.get("n_pseudo_labels", 0),
        }
    except Exception:
        return None


def get_drift_status():
    path = "metrics/drift_report.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        ks = d.get("ks_test", {})
        psi = d.get("psi", {})
        return {
            "timestamp":    d.get("timestamp", ""),
            "n_articles":   d.get("n_articles", 0),
            "avg_conf":     round(d.get("avg_confidence", 0) * 100, 1),
            "ks_stat":      round(ks.get("ks_statistic", 0), 4),
            "ks_p":         round(ks.get("p_value", 0), 4),
            "ks_drift":     ks.get("drift_detected", False),
            "psi":          round(psi.get("value", 0), 4),
            "psi_status":   psi.get("status", "STABLE"),
            "retrain":      d.get("retrain_needed", False),
            "streak":       d.get("low_conf_streak_days", 0),
        }
    except Exception:
        return None


def get_monitor_state():
    path = "metrics/monitor_state.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        last = d.get("last_retrain_triggered", "")
        n_files = len(d.get("processed_files", []))
        return {"last_retrain": last[:10] if last else "Never", "n_files": n_files}
    except Exception:
        return None


@app.get("/")
async def home(request: Request):
    pipeline = get_pipeline_status()
    tweets = get_latest_tweets()
    return templates.TemplateResponse("index.html", {
        "request":  request,
        "last_sync": pipeline["last_sync"],
        "status":   pipeline["status"],
        "region":   "United States",
        "stats":    pipeline,
        "tweets":   tweets,
        "result":   None,
        "headline": None,
        "metrics":  get_model_metrics(),
        "drift":    get_drift_status(),
        "monitor":  get_monitor_state(),
    })


@app.post("/analyze")
async def analyze(request: Request, headline: str = Form(...)):
    prob = _predict_batch([headline])[0]
    conf = prob if prob > 0.5 else 1 - prob
    result = {"verdict": "Real" if prob > 0.5 else "Fake", "conf": f"{conf * 100:.1f}%"}

    pipeline = get_pipeline_status()
    tweets = get_latest_tweets()
    return templates.TemplateResponse("index.html", {
        "request":  request,
        "result":   result,
        "headline": headline,
        "last_sync": pipeline["last_sync"],
        "status":   pipeline["status"],
        "region":   "United States",
        "stats":    pipeline,
        "tweets":   tweets,
        "metrics":  get_model_metrics(),
        "drift":    get_drift_status(),
        "monitor":  get_monitor_state(),
    })
