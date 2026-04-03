import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from scipy import stats
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 100  # must match src/train.py

def clean_text(text):
    import re
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── US political relevancy keywords ───────────────────────────────────────────
US_POLITICAL_KEYWORDS = [
    # US-Iran conflict
    'iran', 'iranian', 'irgc', 'tehran', 'khamenei', 'strait of hormuz',
    'airstrike', 'us strikes', 'nuclear deal', 'sanctions', 'persian gulf',
    'houthis', 'proxy war', 'iran missile', 'iran attack', 'middle east',
    # General US politics
    'congress', 'white house', 'biden', 'trump', 'senate', 'election',
    'democrat', 'republican', 'washington', 'president', 'pentagon',
    'legislation', 'vote', 'ballot', 'campaign', 'policy',
]


# ── Statistical drift detection ────────────────────────────────────────────────

def build_reference_distribution(model, tokenizer, max_len):
    """Compute model prediction probabilities on the training split.
    Saved to models/reference_score_distribution.npy for future runs.
    """
    df = pd.read_csv('data/processed/gossipcop_cleaned.csv').dropna(
        subset=['clean_title', 'label']
    )
    df['label'] = df['label'].astype(int)
    X, y = df['clean_title'].astype(str), df['label']

    # Use training split only — same seed as train.py
    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sequences = tokenizer.texts_to_sequences(X_train)
    padded = pad_sequences(sequences, maxlen=max_len,
                           padding='post', truncating='post')
    ref_scores = model.predict(padded, batch_size=256, verbose=0).flatten()
    np.save('models/reference_score_distribution.npy', ref_scores)
    print(f"Reference distribution built: {len(ref_scores)} samples.")
    return ref_scores


def ks_test(reference_scores, new_scores):
    """Kolmogorov-Smirnov test between reference and incoming score distributions.

    Returns dict with ks_statistic, p_value, and drift_detected (p < 0.05).
    Industry standard: p < 0.05 indicates statistically significant shift.
    """
    stat, p_value = stats.ks_2samp(reference_scores, new_scores)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value":       round(float(p_value), 4),
        "drift_detected": bool(p_value < 0.05),
    }


def compute_psi(reference_scores, new_scores, n_bins=10):
    """Population Stability Index (PSI) between reference and new distributions.

    PSI thresholds (industry standard — cite in thesis):
      PSI < 0.10  → no significant drift (STABLE)
      0.10 ≤ PSI < 0.25 → moderate drift (MONITOR)
      PSI ≥ 0.25  → significant drift (RETRAIN)
    """
    # Bin edges from reference quantiles
    bin_edges = np.percentile(reference_scores, np.linspace(0, 100, n_bins + 1))
    bin_edges[0]  = 0.0
    bin_edges[-1] = 1.0
    bin_edges = np.unique(bin_edges)  # remove duplicates at extremes

    ref_counts, _ = np.histogram(reference_scores, bins=bin_edges)
    new_counts, _ = np.histogram(new_scores,       bins=bin_edges)

    # Convert to fractions; replace zeros to avoid log(0)
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
    print(f"Monitoring on: {latest_file} ({len(df)} articles)")

    # 2. Relevancy check
    df['is_relevant'] = df['text'].str.lower().apply(
        lambda x: any(kw in str(x) for kw in US_POLITICAL_KEYWORDS)
    )
    relevancy_rate = df['is_relevant'].mean()
    print(f"US Political Relevancy Rate: {relevancy_rate:.2%}")

    # 3. Model inference
    model = tf.keras.models.load_model('models/cnn_v1.h5')
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    df['cleaned'] = df['text'].apply(clean_text)
    sequences = tokenizer.texts_to_sequences(df['cleaned'])
    padded = pad_sequences(sequences, maxlen=MAX_LEN,
                           padding='post', truncating='post')
    predictions = model.predict(padded, verbose=0)

    avg_conf = float(np.mean(np.abs(predictions - 0.5)) * 2)
    pred_std  = float(np.std(predictions))
    print(f"Avg decision confidence: {avg_conf:.4f} (threshold: 0.30)")
    print(f"Prediction std-dev:      {pred_std:.4f}")

    # 4. Statistical drift detection (skipped if reference data unavailable)
    ref_path = 'models/reference_score_distribution.npy'
    new_scores = predictions.flatten()
    ks_result = None
    psi_value = None
    psi_status = None

    if not os.path.exists(ref_path):
        processed_path = 'data/processed/gossipcop_cleaned.csv'
        if os.path.exists(processed_path):
            print("Building reference distribution (first run)...")
            ref_scores = build_reference_distribution(model, tokenizer, MAX_LEN)
            ks_result  = ks_test(ref_scores, new_scores)
            psi_value  = compute_psi(ref_scores, new_scores)
            psi_status = _psi_label(psi_value)
        else:
            print("Reference distribution unavailable — skipping KS/PSI tests.")
    else:
        ref_scores = np.load(ref_path)
        print(f"Reference distribution loaded: {len(ref_scores)} samples.")
        ks_result  = ks_test(ref_scores, new_scores)
        psi_value  = compute_psi(ref_scores, new_scores)
        psi_status = _psi_label(psi_value)

    if ks_result:
        print(f"KS-Test  : stat={ks_result['ks_statistic']:.4f}  "
              f"p={ks_result['p_value']:.4f}  drift={ks_result['drift_detected']}")
        print(f"PSI      : {psi_value:.4f} ({psi_status})")

    # 5. Save drift report
    drift_report = {
        "timestamp":      datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_file":    latest_file,
        "n_articles":     int(len(df)),
        "relevancy_rate": round(relevancy_rate, 4),
        "avg_confidence": round(avg_conf, 4),
        "pred_std":       round(pred_std, 4),
        "ks_test":        ks_result,
        "psi": {
            "value":  psi_value,
            "status": psi_status,
            "thresholds": {"stable": 0.10, "moderate": 0.25},
        } if psi_value is not None else None,
    }
    with open('metrics/drift_report.json', 'w') as f:
        json.dump(drift_report, f, indent=2)
    print("Drift report saved to metrics/drift_report.json")

    # 6. Trigger retraining on any drift signal
    drift_triggered = (
        relevancy_rate < 0.30
        or avg_conf < 0.30
        or (ks_result is not None and ks_result['drift_detected'])
        or (psi_value is not None and psi_value >= 0.25)
    )

    if drift_triggered:
        reasons = []
        if relevancy_rate < 0.30:
            reasons.append(f"low relevancy ({relevancy_rate:.2%})")
        if avg_conf < 0.30:
            reasons.append(f"low confidence ({avg_conf:.4f})")
        if ks_result and ks_result['drift_detected']:
            reasons.append(f"KS p={ks_result['p_value']:.4f}")
        if psi_value is not None and psi_value >= 0.25:
            reasons.append(f"PSI={psi_value:.4f}")
        print(f"WARNING: Drift detected — {', '.join(reasons)}. Triggering retraining.")
        sys.exit(1)

    print("System Stable: Model performing well on current US news.")
    sys.exit(0)


if __name__ == "__main__":
    run_monitoring()
