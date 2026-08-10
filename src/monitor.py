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
    """Build reference distribution from the EARLIEST live scraped data.

    Why live data instead of PHEME?
    BERTweet was fine-tuned on PHEME (UK breaking-news tweets, 2014-2017).
    Even for real-class PHEME examples the model assigns ~10% of them low
    fake-probability scores (it's uncertain on hard PHEME examples). This
    creates a *bimodal* reference (10% near 0, 89% near 1). Live 2026 US
    political tweets get *all* scored near 0.99 (low std), so the bin
    [0–0.1] has 10.4% in reference vs ~0% in live → PSI ≫ 0.25 every day,
    even when nothing has changed.

    Fix: use the EARLIEST scraped files as the "clean baseline". Those were
    collected at deployment time, before any concept drift, and come from the
    same domain (2026 US political Twitter). Reference and live data then
    share the same distribution shape, so PSI only rises when something real
    has shifted.

    Falls back to PHEME real-class if no live data is available yet.
    """
    data_dir = 'data/new_scraped/'
    # Sort by filename (encodes date YYYYMMDD) so earliest files come first;
    # skip the accumulation file and any archive sub-directory entries.
    csv_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.csv') and f != 'all_tweets.csv'
    ])

    if csv_files:
        # Use the 10 oldest files as the clean baseline
        baseline_files = csv_files[:10]
        dfs = []
        for fname in baseline_files:
            df_b = pd.read_csv(os.path.join(data_dir, fname))
            if 'text' in df_b.columns:
                dfs.append(df_b[['text']])
        if dfs:
            df_baseline = pd.concat(dfs, ignore_index=True).drop_duplicates('text')
            texts = df_baseline['text'].apply(normalise_tweet).tolist()[:2000]
            ref_scores = predict_bertweet(model, tokenizer, device, texts)
            np.save('models/reference_score_distribution.npy', ref_scores)
            print(
                f"Reference distribution built from {len(ref_scores)} early live tweets "
                f"({len(baseline_files)} files: {baseline_files[0]} … {baseline_files[-1]})."
            )
            return ref_scores

    # ── Fallback: PHEME real-class only (if no live data yet) ────────────────
    print("No live scraped data for reference — falling back to PHEME real-class.")
    df = pd.read_csv('data/processed/pheme_cleaned.csv').dropna(
        subset=['clean_title', 'label']
    )
    df['label'] = df['label'].astype(int)
    df_real = df[df['label'] == 1]
    X_train, _, _, _ = train_test_split(
        df_real['clean_title'].astype(str), df_real['label'],
        test_size=0.2, random_state=42
    )
    ref_texts = X_train.tolist()[:2000]
    ref_scores = predict_bertweet(model, tokenizer, device, ref_texts)
    np.save('models/reference_score_distribution.npy', ref_scores)
    print(f"Reference distribution built from {len(ref_scores)} real-class PHEME examples.")
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


# ── Monitor state (processed-file tracking + retraining cooldown) ─────────────

MONITOR_STATE_PATH = 'metrics/monitor_state.json'
RETRAIN_COOLDOWN_DAYS = 30   # don't retrain more often than once a month


def load_monitor_state():
    if os.path.exists(MONITOR_STATE_PATH):
        with open(MONITOR_STATE_PATH) as f:
            return json.load(f)
    return {'processed_files': [], 'last_retrain_triggered': None}


def save_monitor_state(state):
    os.makedirs('metrics', exist_ok=True)
    with open(MONITOR_STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)


def is_in_retrain_cooldown(state):
    last = state.get('last_retrain_triggered')
    if not last:
        return False
    last_dt = datetime.fromisoformat(last.replace('Z', ''))
    return (datetime.utcnow() - last_dt).days < RETRAIN_COOLDOWN_DAYS


# ── Main monitoring loop ───────────────────────────────────────────────────────

def run_monitoring():
    os.makedirs('metrics', exist_ok=True)

    # 1. Load latest scraped data
    data_dir = 'data/new_scraped/'
    if not os.path.exists(data_dir):
        print("Scraped data directory not found — skipping monitoring.")
        sys.exit(0)

    # Sort by FILENAME (date is encoded as YYYYMMDD in the name) so ordering is
    # stable across fresh git checkouts where all files share the same mtime.
    # Exclude the cumulative all_tweets.csv — we monitor per-batch files only.
    csv_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.csv') and f != 'all_tweets.csv'
    ])
    if not csv_files:
        print("No scraped data found — skipping monitoring.")
        sys.exit(0)

    # Skip files we've already processed to avoid re-triggering drift on the
    # same data every day (critical when the scraper has stopped producing
    # new files, e.g. when X_BEARER_TOKEN expires).
    state = load_monitor_state()
    processed = set(state.get('processed_files', []))

    unprocessed = [f for f in csv_files if f not in processed]
    if not unprocessed:
        print("All scraped files have already been processed — nothing new to monitor.")
        print(f"  Total processed: {len(processed)} files, latest: {csv_files[-1]}")
        print("  If the scraper has stopped, check that X_BEARER_TOKEN is valid.")
        sys.exit(0)

    # Pick the newest unprocessed file
    latest_fname = unprocessed[-1]
    latest_file  = os.path.join(data_dir, latest_fname)

    df = pd.read_csv(latest_file)
    # Strip any OOD injection rows (source == "drift_injection") so they
    # never pollute a real monitoring run after the test is done.
    if 'source' in df.columns:
        n_before = len(df)
        df = df[df['source'] != 'drift_injection'].reset_index(drop=True)
        if len(df) < n_before:
            print(f"Filtered {n_before - len(df)} OOD injection rows from stream.")
    print(f"Monitoring on: {latest_file} ({len(df)} tweets)  "
          f"[{len(unprocessed)-1} other unprocessed files queued]")

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
    # load_bertweet() handles local → HF Hub → base model fallback automatically
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

    def _reference_is_stale(scores):
        """Detect the PHEME-bimodal reference bug.

        A reference built from mixed PHEME examples has ~10% of values near 0
        (fake predictions) and ~89% near 1 (real predictions). Live 2026
        political tweets are almost all near 1, so the [0-0.1] bin mismatch
        drives PSI to 5+ every day.

        A valid live-data reference should have < 5% of values below 0.5.
        """
        frac_below_half = float(np.mean(scores < 0.5))
        if frac_below_half > 0.05:
            print(
                f"  WARNING: reference looks stale/bimodal — "
                f"{frac_below_half:.1%} of values below 0.5. Rebuilding from live data."
            )
            return True
        return False

    if not os.path.exists(ref_path):
        print("Reference distribution missing — building from live scraped data...")
        ref_scores = build_reference_distribution(model, tokenizer, device)
    else:
        ref_scores = np.load(ref_path)
        print(f"Reference distribution loaded: {len(ref_scores)} samples.")
        if _reference_is_stale(ref_scores):
            ref_scores = build_reference_distribution(model, tokenizer, device)

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

    # 8. Mark this file as processed so we don't re-analyse it tomorrow.
    # This replaces the broken os.rename() archive: renaming on a GitHub
    # Actions runner is pointless because git checkout restores every file
    # on the next run. Instead, we track which files have been processed in
    # a JSON state file that IS committed to git.
    state['processed_files'] = sorted(set(processed) | {latest_fname})
    # Trim to last 120 entries (4 months) to keep the file small
    state['processed_files'] = state['processed_files'][-120:]

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

        # Retraining cooldown — prevent triggering more than once per month.
        # Without this, every daily monitoring run after a drift event would
        # kick off a fresh GPU retraining job while the first one is still
        # running (or hasn't been deployed yet).
        if is_in_retrain_cooldown(state):
            last_rt = state.get('last_retrain_triggered', 'unknown')
            days_since = (datetime.utcnow() - datetime.fromisoformat(last_rt.replace('Z', ''))).days
            print(
                f"\n⏳ Retraining cooldown active — last triggered {days_since}d ago "
                f"(cooldown: {RETRAIN_COOLDOWN_DAYS}d). Skipping retrain trigger."
            )
            save_monitor_state(state)
            sys.exit(1)   # still exit(1) so GitHub output = drift=true, but no retrain

        print("\n[Pseudo-Labeling] Generating labels for accumulated tweets...")
        pseudo_df = generate_pseudo_labels(model, tokenizer, device)

        if len(pseudo_df) > 0:
            augmented_path = create_augmented_dataset(pseudo_df)
            print(f"\n✓ Augmented dataset ready: {augmented_path}")
            print("✓ Triggering GPU retraining workflow...")
        else:
            print("\n⚠ Not enough confident predictions for pseudo-labeling.")
            print("  Retraining will proceed on PHEME only.")

        state['last_retrain_triggered'] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        save_monitor_state(state)
        sys.exit(1)

    save_monitor_state(state)
    print("System STABLE: model performing well on current tweet stream.")
    sys.exit(0)


if __name__ == "__main__":
    run_monitoring()
