import os
import re
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, date
from scipy import stats
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

# ── Tweet normalisation (must match PHEME preprocessing in download_pheme.py) ──

def normalise_tweet(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "HTTPURL", text)
    text = re.sub(r"@\w+", "@USER", text)
    return re.sub(r"\s+", " ", text).strip()


# ── US political relevancy keywords ───────────────────────────────────────────

US_POLITICAL_KEYWORDS = [
    'iran', 'iranian', 'irgc', 'tehran', 'khamenei', 'strait of hormuz',
    'airstrike', 'us strikes', 'nuclear deal', 'sanctions', 'persian gulf',
    'houthis', 'proxy war', 'iran missile', 'iran attack', 'middle east',
    'congress', 'white house', 'biden', 'trump', 'senate', 'election',
    'democrat', 'republican', 'washington', 'president', 'pentagon',
    'legislation', 'vote', 'ballot', 'campaign', 'policy',
]


# ── BERTweet model loading & inference ───────────────────────────────────────

def load_bertweet():
    import torch, yaml
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    with open('params.yaml') as f:
        model_name = yaml.safe_load(f)['bertweet']['model_name']
    model_dir = 'models/bertweet_finetuned'
    # Load tokenizer from HF cache (guaranteed to have bpe.codes)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # Load weights: fine-tuned local → HF Hub → base model (fallback)
    hf_hub = "abdellahzegdane/fakeshield-bertweet"
    if os.path.exists(model_dir):
        print(f"Loading fine-tuned model from {model_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    else:
        try:
            print(f"Local model not found — downloading from HF Hub: {hf_hub}")
            model = AutoModelForSequenceClassification.from_pretrained(hf_hub)
            print("HF Hub model loaded successfully.")
        except Exception as e:
            print(f"HF Hub failed ({e}) — falling back to base model {model_name}")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    return model, tokenizer, device


def predict_bertweet(model, tokenizer, device, texts, batch_size=64, max_len=128):
    import torch
    all_probs = []
    texts = list(texts)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, max_length=max_len, padding=True,
            truncation=True, return_tensors='pt'
        )
        with torch.no_grad():
            logits = model(
                input_ids=enc['input_ids'].to(device),
                attention_mask=enc['attention_mask'].to(device),
            ).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
    return np.array(all_probs)


# ── Reference distribution ────────────────────────────────────────────────────

def build_reference_distribution(model, tokenizer, device):
    """Compute BERTweet predictions on the PHEME training split and cache them."""
    df = pd.read_csv('data/processed/pheme_cleaned.csv').dropna(
        subset=['clean_title', 'label']
    )
    df['label'] = df['label'].astype(int)
    X, y = df['clean_title'].astype(str), df['label']
    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    ref_scores = predict_bertweet(model, tokenizer, device, X_train.tolist())
    np.save('models/reference_score_distribution.npy', ref_scores)
    print(f"Reference distribution built: {len(ref_scores)} samples.")
    return ref_scores


# ── Statistical drift tests ───────────────────────────────────────────────────

def ks_test(reference_scores, new_scores):
    stat, p_value = stats.ks_2samp(reference_scores, new_scores)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value":       round(float(p_value), 4),
        "drift_detected": bool(p_value < 0.05),
    }


def compute_psi(reference_scores, new_scores, n_bins=10):
    """Population Stability Index.
    PSI < 0.10 → STABLE | 0.10–0.25 → MODERATE | ≥ 0.25 → SIGNIFICANT
    """
    bin_edges = np.percentile(reference_scores, np.linspace(0, 100, n_bins + 1))
    bin_edges[0]  = 0.0
    bin_edges[-1] = 1.0
    bin_edges = np.unique(bin_edges)

    ref_counts, _ = np.histogram(reference_scores, bins=bin_edges)
    new_counts, _ = np.histogram(new_scores,       bins=bin_edges)

    eps = 1e-4
    ref_frac = np.maximum(ref_counts / len(reference_scores), eps)
    new_frac = np.maximum(new_counts / len(new_scores),       eps)
    psi = float(np.sum((new_frac - ref_frac) * np.log(new_frac / ref_frac)))
    return round(psi, 4)


def _psi_label(psi):
    if psi < 0.10:
        return "STABLE"
    elif psi < 0.25:
        return "MODERATE"
    return "SIGNIFICANT"


# ── Consecutive low-confidence tracking (3-day rule) ─────────────────────────

CONFIDENCE_HISTORY_PATH = 'metrics/confidence_history.json'


def load_confidence_history():
    if os.path.exists(CONFIDENCE_HISTORY_PATH):
        with open(CONFIDENCE_HISTORY_PATH) as f:
            return json.load(f).get('history', [])
    return []


def update_confidence_history(avg_conf):
    history = load_confidence_history()
    today = date.today().isoformat()
    # Replace today's entry if already present, otherwise append
    history = [h for h in history if h['date'] != today]
    history.append({'date': today, 'avg_confidence': round(avg_conf, 4)})
    # Keep only last 30 days
    history = sorted(history, key=lambda h: h['date'])[-30:]
    os.makedirs('metrics', exist_ok=True)
    with open(CONFIDENCE_HISTORY_PATH, 'w') as f:
        json.dump({'history': history}, f, indent=2)
    return history


def sustained_low_confidence(history, threshold=0.30, days=3):
    """True if the last `days` consecutive entries all have confidence < threshold."""
    recent = sorted(history, key=lambda h: h['date'])[-days:]
    if len(recent) < days:
        return False
    return all(h['avg_confidence'] < threshold for h in recent)


# ── Pseudo-labeling for retraining ────────────────────────────────────────────

def generate_pseudo_labels(model, tokenizer, device, confidence_threshold=0.3):
    data_dir = 'data/new_scraped/'
    if not os.path.exists(data_dir):
        print("No scraped data found for pseudo-labeling.")
        return pd.DataFrame()

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print("No scraped CSV files found.")
        return pd.DataFrame()

    df_scraped = pd.concat(
        [pd.read_csv(os.path.join(data_dir, f)) for f in csv_files],
        ignore_index=True
    ).drop_duplicates(subset=['text'])
    print(f"Loaded {len(df_scraped)} unique scraped tweets for pseudo-labeling.")

    # Normalise exactly as PHEME preprocessing does
    df_scraped['clean_title'] = df_scraped['text'].apply(normalise_tweet)

    predictions = predict_bertweet(
        model, tokenizer, device, df_scraped['clean_title'].tolist()
    )

    confidence_scores = np.abs(predictions - 0.5)
    confident_mask = confidence_scores > confidence_threshold

    df_confident = df_scraped[confident_mask].copy()
    df_confident['label']      = (predictions[confident_mask] > 0.5).astype(int)
    df_confident['confidence'] = confidence_scores[confident_mask]
    df_confident['source']     = 'pseudo_label'

    n_fake = (df_confident['label'] == 0).sum()
    n_real = (df_confident['label'] == 1).sum()
    print(f"Pseudo-labeled {len(df_confident)} confident tweets:")
    print(f"  - {n_fake} FAKE  (p < {0.5 - confidence_threshold:.2f})")
    print(f"  - {n_real} REAL  (p > {0.5 + confidence_threshold:.2f})")
    print(f"  - Discarded {(~confident_mask).sum()} uncertain predictions")

    return df_confident[['clean_title', 'label', 'confidence', 'source']]


def create_augmented_dataset(pseudo_df, output_path='data/processed/pheme_augmented.csv'):
    df_pheme = pd.read_csv('data/processed/pheme_cleaned.csv').dropna(
        subset=['clean_title', 'label']
    )
    df_pheme['label']      = df_pheme['label'].astype(int)
    df_pheme['source']     = 'ground_truth'
    df_pheme['confidence'] = 1.0

    df_combined = pd.concat([
        df_pheme[['clean_title', 'label', 'source', 'confidence']],
        pseudo_df[['clean_title', 'label', 'source', 'confidence']],
    ], ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_combined.to_csv(output_path, index=False)

    n_pheme  = len(df_pheme)
    n_pseudo = len(pseudo_df)
    n_total  = len(df_combined)
    print(f"\nAugmented dataset created:")
    print(f"  PHEME (ground truth): {n_pheme}  ({n_pheme/n_total*100:.1f}%)")
    print(f"  Pseudo-labeled:       {n_pseudo} ({n_pseudo/n_total*100:.1f}%)")
    print(f"  Total:                {n_total}")

    metadata = {
        "created_at":      datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_ground_truth":  int(n_pheme),
        "n_pseudo_labeled": int(n_pseudo),
        "n_total":         int(n_total),
        "pseudo_label_ratio": round(n_pseudo / n_total, 4),
        "confidence_threshold": 0.3,
    }
    with open('metrics/augmented_dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return output_path


# ── Main monitoring loop ───────────────────────────────────────────────────────

def run_monitoring():
    os.makedirs('metrics', exist_ok=True)

    # 1. Load latest scraped data
    data_dir = 'data/new_scraped/'
    if not os.path.exists(data_dir):
        print("Scraped data directory not found — skipping monitoring.")
        sys.exit(0)

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print("No scraped data found — skipping monitoring.")
        sys.exit(0)

    latest_file = max(
        [os.path.join(data_dir, f) for f in csv_files],
        key=os.path.getmtime
    )
    df = pd.read_csv(latest_file)
    print(f"Monitoring on: {latest_file} ({len(df)} tweets)")

    # 2. Relevancy check — diagnostic only, does NOT trigger retraining
    df['is_relevant'] = df['text'].str.lower().apply(
        lambda x: any(kw in str(x) for kw in US_POLITICAL_KEYWORDS)
    )
    relevancy_rate = df['is_relevant'].mean()
    print(f"US Political Relevancy Rate: {relevancy_rate:.2%}  (diagnostic only)")
    if relevancy_rate < 0.30:
        print("  WARNING: low relevancy — possible scraper data-quality issue, "
              "not a drift trigger.")

    # 3. BERTweet inference
    model_dir = 'models/bertweet_finetuned'
    if not os.path.exists(model_dir):
        print("BERTweet model not found — skipping monitoring.")
        sys.exit(0)

    model, tokenizer, device = load_bertweet()
    print(f"BERTweet loaded on {device}")

    df['clean'] = df['text'].apply(normalise_tweet)
    predictions = predict_bertweet(model, tokenizer, device, df['clean'].tolist())

    # conf = mean |p - 0.5|  (range 0–0.5; high means decisive)
    avg_conf = float(np.mean(np.abs(predictions - 0.5)))
    pred_std  = float(np.std(predictions))
    print(f"Avg decision confidence: {avg_conf:.4f}  (threshold: 0.15 ≈ 0.30 scaled)")
    print(f"Prediction std-dev:      {pred_std:.4f}")

    # 4. Update 3-day confidence history
    history = update_confidence_history(avg_conf)
    low_conf_streak = sustained_low_confidence(history, threshold=0.15, days=3)

    # 5. Statistical drift detection
    ref_path   = 'models/reference_score_distribution.npy'
    new_scores = predictions
    ks_result  = None
    psi_value  = None
    psi_status = None

    processed_path = 'data/processed/pheme_cleaned.csv'
    if not os.path.exists(ref_path):
        if os.path.exists(processed_path):
            print("Building reference distribution (first run)...")
            ref_scores = build_reference_distribution(model, tokenizer, device)
        else:
            print("Reference distribution unavailable — skipping KS/PSI tests.")
            ref_scores = None
    else:
        ref_scores = np.load(ref_path)
        print(f"Reference distribution loaded: {len(ref_scores)} samples.")

    if ref_scores is not None:
        ks_result  = ks_test(ref_scores, new_scores)
        psi_value  = compute_psi(ref_scores, new_scores)
        psi_status = _psi_label(psi_value)
        print(f"KS-Test : stat={ks_result['ks_statistic']:.4f}  "
              f"p={ks_result['p_value']:.4f}  drift={ks_result['drift_detected']}")
        print(f"PSI     : {psi_value:.4f} ({psi_status})")

    # 6. Retraining decision — three conditions from the paper (Section 7.2):
    #    (a) PSI ≥ 0.25
    #    (b) KS p < 0.05
    #    (c) avg confidence < 0.15 for 3+ consecutive days
    drift_triggered = (
        (psi_value  is not None and psi_value >= 0.25)
        or (ks_result is not None and ks_result['drift_detected'])
        or low_conf_streak
    )

    # 7. Save drift report
    drift_report = {
        "timestamp":      datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_file":    latest_file,
        "n_articles":     int(len(df)),
        "relevancy_rate": round(relevancy_rate, 4),
        "avg_confidence": round(avg_conf, 4),
        "pred_std":       round(pred_std, 4),
        "low_conf_streak_days": sum(
            1 for h in sorted(history, key=lambda x: x['date'])[-3:]
            if h['avg_confidence'] < 0.15
        ),
        "ks_test": ks_result,
        "psi": {
            "value":  psi_value,
            "status": psi_status,
            "thresholds": {"stable": 0.10, "moderate": 0.25},
        } if psi_value is not None else None,
        "retrain_needed": drift_triggered,
    }
    with open('metrics/drift_report.json', 'w') as f:
        json.dump(drift_report, f, indent=2)
    print("Drift report saved to metrics/drift_report.json")

    if drift_triggered:
        reasons = []
        if psi_value is not None and psi_value >= 0.25:
            reasons.append(f"PSI={psi_value:.4f}")
        if ks_result and ks_result['drift_detected']:
            reasons.append(f"KS p={ks_result['p_value']:.4f}")
        if low_conf_streak:
            reasons.append("3-day low confidence streak")

        print(f"\n{'='*70}")
        print(f"WARNING: Drift detected — {', '.join(reasons)}")
        print(f"{'='*70}")

        print("\n[Pseudo-Labeling] Generating labels for accumulated tweets...")
        pseudo_df = generate_pseudo_labels(model, tokenizer, device)

        if len(pseudo_df) > 0:
            augmented_path = create_augmented_dataset(pseudo_df)
            print(f"\n✓ Augmented dataset ready: {augmented_path}")
            print("✓ Triggering GPU retraining workflow...")
        else:
            print("\n⚠ Not enough confident predictions for pseudo-labeling.")
            print("  Retraining will proceed on PHEME only.")

        sys.exit(1)

    print("System STABLE: model performing well on current tweet stream.")
    sys.exit(0)


if __name__ == "__main__":
    run_monitoring()
