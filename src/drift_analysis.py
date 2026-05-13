"""
Drift detector validation.

Answers the advisor's question: what is the relationship between the
drift detectors (PSI, KS) and the actual distribution shift?

Method — controlled mixing experiment:
  The PHEME test split represents the reference distribution (in-distribution).
  The accumulated live tweets represent a shifted distribution (out-of-distribution).

  We build 11 synthetic batches by mixing these two sources at mixing ratios
  α ∈ {0%, 10%, 20%, ..., 100%}, where α = fraction of live tweets.

  At α=0   the batch is pure PHEME → detectors should read near zero.
  At α=100 the batch is pure live tweets → detectors should read maximum shift.

  For each α we compute:
    - PSI against the PHEME reference distribution
    - KS statistic and p-value against the PHEME reference distribution
    - Avg model confidence (lower = more uncertain = more shifted)

  Results saved to metrics/drift_analysis.json for inclusion in the paper.

Requirements:
  - models/bertweet_finetuned/   (production BERTweet)
  - data/processed/pheme_cleaned.csv
  - data/new_scraped/*.csv       (accumulated live tweets)
"""

import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)


# ── Tweet normalisation ────────────────────────────────────────────────────────

def normalise_tweet(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "HTTPURL", text)
    text = re.sub(r"@\w+", "@USER", text)
    return re.sub(r"\s+", " ", text).strip()


# ── BERTweet inference ────────────────────────────────────────────────────────

def load_bertweet():
    import torch, yaml
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_dir = 'models/bertweet_finetuned'
    # Load tokenizer from the original HF cache (vinai/bertweet-base), NOT from
    # model_dir. The saved model_dir may be missing bpe.codes which BertweetTokenizer
    # requires — loading from HF cache guarantees all files are present.
    with open('params.yaml') as f:
        model_name = yaml.safe_load(f)['bertweet']['model_name']
    print(f"Loading tokenizer from HF cache: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    print(f"Tokenizer: {tokenizer.__class__.__name__}", flush=True)
    print(f"Loading model weights from {model_dir}", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Moving model to {device}...", flush=True)
    model = model.to(device).eval()
    print("BERTweet ready.", flush=True)
    return model, tokenizer, device


def predict_proba(model, tokenizer, device, texts, batch_size=64, max_len=128):
    import torch
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        enc = tokenizer(batch, max_length=max_len, padding=True,
                        truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = model(
                input_ids=enc['input_ids'].to(device),
                attention_mask=enc['attention_mask'].to(device),
            ).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
    return np.array(all_probs)


# ── Drift statistics ───────────────────────────────────────────────────────────

def compute_psi(reference, actual, n_bins=10):
    bin_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bin_edges[0]  = 0.0
    bin_edges[-1] = 1.0
    bin_edges = np.unique(bin_edges)
    eps = 1e-4
    ref_frac = np.maximum(
        np.histogram(reference, bins=bin_edges)[0] / len(reference), eps
    )
    act_frac = np.maximum(
        np.histogram(actual,    bins=bin_edges)[0] / len(actual),    eps
    )
    return float(np.sum((act_frac - ref_frac) * np.log(act_frac / ref_frac)))


def psi_label(psi):
    if psi < 0.10:
        return "STABLE"
    elif psi < 0.25:
        return "MODERATE"
    return "SIGNIFICANT"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs('metrics', exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    model_dir = 'models/bertweet_finetuned'
    if not os.path.exists(model_dir):
        print("BERTweet model not found — skipping drift analysis.")
        return

    print("Loading BERTweet...")
    model, tokenizer, device = load_bertweet()

    # PHEME test split (reference domain)
    df_pheme = pd.read_csv('data/processed/pheme_cleaned.csv').dropna(
        subset=['clean_title', 'label']
    )
    df_pheme['label'] = df_pheme['label'].astype(int)
    X_pheme, y_pheme = df_pheme['clean_title'].astype(str), df_pheme['label']
    _, X_pheme_test, _, _ = train_test_split(
        X_pheme, y_pheme, test_size=0.2, random_state=42, stratify=y_pheme
    )
    X_pheme_test = X_pheme_test.tolist()
    print(f"PHEME test split: {len(X_pheme_test)} tweets")

    # Live tweets (shifted domain)
    data_dir = 'data/new_scraped/'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    df_live = pd.concat(
        [pd.read_csv(os.path.join(data_dir, f)) for f in csv_files],
        ignore_index=True
    ).drop_duplicates(subset=['text'])
    X_live = df_live['text'].apply(normalise_tweet).tolist()
    print(f"Live tweets: {len(X_live)} unique")

    # ── Reference distribution: BERTweet scores on PHEME training split ───────
    df_train_ref = df_pheme.copy()
    X_train_ref, _, _, _ = train_test_split(
        df_pheme['clean_title'].astype(str), df_pheme['label'],
        test_size=0.2, random_state=42, stratify=df_pheme['label']
    )
    print("\nComputing reference distribution (PHEME train split)...")
    ref_scores = predict_proba(model, tokenizer, device, X_train_ref.tolist())
    print(f"  ref mean={ref_scores.mean():.4f}  std={ref_scores.std():.4f}")

    # ── Mixing experiment ──────────────────────────────────────────────────────
    mixing_ratios = [round(a / 10, 1) for a in range(0, 11)]  # 0.0 to 1.0
    np.random.seed(42)
    n_per_batch = min(len(X_pheme_test), len(X_live), 200)

    results = []
    print(f"\nRunning mixing experiment (batch size={n_per_batch} per ratio)...")
    print(f"{'α (live %)':<12} {'PSI':>8} {'Status':<12} {'KS-stat':>9} {'KS-p':>8} {'Avg Conf':>10}")
    print("-" * 65)

    for alpha in mixing_ratios:
        n_live  = int(round(n_per_batch * alpha))
        n_pheme = n_per_batch - n_live

        # Sample without replacement (or with if needed)
        pheme_sample = list(np.random.choice(X_pheme_test, size=n_pheme, replace=n_pheme > len(X_pheme_test)))
        live_sample  = list(np.random.choice(X_live,       size=n_live,  replace=n_live  > len(X_live)))       if n_live > 0 else []

        batch = pheme_sample + live_sample
        np.random.shuffle(batch)

        scores = predict_proba(model, tokenizer, device, batch)

        psi_val  = compute_psi(ref_scores, scores)
        ks_stat, ks_p = stats.ks_2samp(ref_scores, scores)
        avg_conf = float(np.mean(np.abs(scores - 0.5)))

        row = {
            "alpha":          alpha,
            "n_live":         n_live,
            "n_pheme":        n_pheme,
            "psi":            round(psi_val, 4),
            "psi_status":     psi_label(psi_val),
            "ks_statistic":   round(float(ks_stat), 4),
            "ks_p_value":     round(float(ks_p), 4),
            "ks_drift":       bool(ks_p < 0.05),
            "avg_confidence": round(avg_conf, 4),
        }
        results.append(row)
        print(f"{alpha*100:>8.0f}%    {psi_val:>8.4f} {psi_label(psi_val):<12} "
              f"{ks_stat:>9.4f} {ks_p:>8.4f} {avg_conf:>10.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    output = {
        "description": (
            "Controlled mixing experiment: α=fraction of live tweets in batch. "
            "PSI and KS are computed against the BERTweet reference distribution "
            "built from the PHEME training split."
        ),
        "n_per_batch":      n_per_batch,
        "n_pheme_test":     len(X_pheme_test),
        "n_live_tweets":    len(X_live),
        "thresholds": {
            "psi_moderate":   0.10,
            "psi_significant": 0.25,
            "ks_drift_p":     0.05,
            "confidence_low": 0.15,
        },
        "results": results,
    }
    with open('metrics/drift_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved: metrics/drift_analysis.json")

    # ── Key finding summary ───────────────────────────────────────────────────
    psi_trigger_alpha = next(
        (r['alpha'] for r in results if r['psi'] >= 0.25), None
    )
    ks_trigger_alpha = next(
        (r['alpha'] for r in results if r['ks_drift']), None
    )
    print(f"\nKey findings:")
    if psi_trigger_alpha is not None:
        print(f"  PSI triggers (≥0.25) at α={psi_trigger_alpha:.0%} live tweets")
    else:
        print("  PSI never reaches SIGNIFICANT threshold in this experiment")
    if ks_trigger_alpha is not None:
        print(f"  KS triggers (p<0.05) at α={ks_trigger_alpha:.0%} live tweets")
    else:
        print("  KS never reaches significance threshold in this experiment")


if __name__ == "__main__":
    main()
