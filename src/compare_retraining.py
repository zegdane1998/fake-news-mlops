"""
Retraining comparison experiment.

Answers the advisor's core question:
  Does retraining on PHEME + pseudo-labeled live tweets improve over
  retraining on PHEME alone?

Procedure:
  1. Load the PHEME-only BERTweet checkpoint  (models/bertweet_pheme_only/)
  2. Use it to pseudo-label accumulated live tweets
  3. Train a second BERTweet on PHEME + pseudo-labels  (models/bertweet_augmented/)
  4. Evaluate BOTH models on the same held-out test split
  5. Save side-by-side metrics to metrics/retraining_comparison.json

Both training runs use identical hyperparameters (params.yaml).
The test split is the PHEME 20% holdout (seed 42, stratified) — the same
split used throughout the paper so numbers are directly comparable.
"""

import json
import os
import re

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

with open("params.yaml") as f:
    P = yaml.safe_load(f)["bertweet"]

MODEL_NAME   = P["model_name"]
MAX_LEN      = int(P["max_len"])
BATCH_SIZE   = int(P["batch_size"])
EPOCHS       = int(P["epochs"])
LR           = float(P["learning_rate"])
WARMUP_RATIO = float(P["warmup_ratio"])
WEIGHT_DECAY = float(P["weight_decay"])
RANDOM_STATE = int(P["random_state"])

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found — this script must run on the Vast.ai instance")
DEVICE = "cuda"
print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0)})")


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalise_tweet(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "HTTPURL", text)
    text = re.sub(r"@\w+", "@USER", text)
    return re.sub(r"\s+", " ", text).strip()


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(
            list(texts), max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            labels=batch["labels"].to(DEVICE),
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += out.loss.item()
    return total_loss / len(loader)


def eval_model(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )
            probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def metrics_dict(y_true, y_preds, y_probs):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_preds)), 4),
        "f1_macro": round(float(f1_score(y_true, y_preds, average="macro")), 4),
        "f1_fake":  round(float(f1_score(y_true, y_preds, pos_label=0)), 4),
        "f1_real":  round(float(f1_score(y_true, y_preds, pos_label=1)), 4),
        "auc_roc":  round(float(roc_auc_score(y_true, y_probs)), 4),
    }


def fine_tune(run_name, X_train, y_train, X_test, y_test,
              tokenizer, save_dir, n_pseudo=0):
    """Fine-tune BERTweet and return test metrics."""
    os.makedirs(save_dir, exist_ok=True)
    mlflow.set_experiment("retraining-comparison")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(P)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_pseudo", n_pseudo)

        train_ds = TweetDataset(X_train, y_train, tokenizer)
        test_ds  = TweetDataset(X_test,  y_test,  tokenizer)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        ).to(DEVICE)

        optimizer     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        total_steps   = len(train_loader) * EPOCHS
        warmup_steps  = int(total_steps * WARMUP_RATIO)
        scheduler     = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_f1, best_state = 0.0, None
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler)
            preds, probs, labels = eval_model(model, test_loader)
            f1 = f1_score(labels, preds, average="macro")
            auc = roc_auc_score(labels, probs)
            print(f"  Epoch {epoch}/{EPOCHS} — loss={train_loss:.4f}  f1={f1:.4f}  auc={auc:.4f}")
            mlflow.log_metrics({"train_loss": train_loss, "f1_macro": f1, "auc_roc": auc}, step=epoch)
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        preds, probs, labels = eval_model(model, test_loader)
        m = metrics_dict(labels, preds, probs)
        print(classification_report(labels, preds, target_names=["fake", "real"]))
        mlflow.log_metrics(m)
        return m


# ── Pseudo-labeling ───────────────────────────────────────────────────────────

def generate_pseudo_labels(model_dir, confidence_threshold=0.3):
    """Use the PHEME-only model to label accumulated live tweets."""
    print(f"\n[Pseudo-Labeling] Loading model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE).eval()

    data_dir = 'data/new_scraped/'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    df_scraped = pd.concat(
        [pd.read_csv(os.path.join(data_dir, f)) for f in csv_files],
        ignore_index=True
    ).drop_duplicates(subset=['text'])
    print(f"Loaded {len(df_scraped)} unique scraped tweets")

    df_scraped['clean_title'] = df_scraped['text'].apply(normalise_tweet)
    texts = df_scraped['clean_title'].tolist()

    all_probs = []
    for i in range(0, len(texts), 64):
        batch = texts[i:i + 64]
        enc = tokenizer(batch, max_length=MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = model(
                input_ids=enc['input_ids'].to(DEVICE),
                attention_mask=enc['attention_mask'].to(DEVICE),
            ).logits
            all_probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist())
    predictions = np.array(all_probs)

    confidence  = np.abs(predictions - 0.5)
    mask        = confidence > confidence_threshold
    df_out      = df_scraped[mask].copy()
    df_out['label']      = (predictions[mask] > 0.5).astype(int)
    df_out['confidence'] = confidence[mask]
    df_out['source']     = 'pseudo_label'

    n_fake = (df_out['label'] == 0).sum()
    n_real = (df_out['label'] == 1).sum()
    print(f"Kept {len(df_out)} confident tweets  ({n_fake} fake / {n_real} real)")
    print(f"Discarded {(~mask).sum()} uncertain predictions")

    os.makedirs('metrics', exist_ok=True)
    with open('metrics/pseudo_label_stats.json', 'w') as f:
        json.dump({
            "total_scraped":       int(len(df_scraped)),
            "n_pseudo_labeled":    int(len(df_out)),
            "n_fake":              int(n_fake),
            "n_real":              int(n_real),
            "n_discarded":         int((~mask).sum()),
            "confidence_threshold": confidence_threshold,
        }, f, indent=2)

    return df_out[['clean_title', 'label', 'confidence', 'source']]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    # ── Shared test split from PHEME ──────────────────────────────────────────
    df_pheme = pd.read_csv('data/processed/pheme_cleaned.csv').dropna(
        subset=['clean_title', 'label']
    )
    df_pheme['label'] = df_pheme['label'].astype(int)
    X_all, y_all = df_pheme['clean_title'].astype(str).tolist(), df_pheme['label'].values

    X_train_pheme, X_test, y_train_pheme, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
    )
    print(f"\nShared test set: {len(X_test)} tweets  "
          f"({(y_test==0).sum()} fake / {(y_test==1).sum()} real)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    # ── Run A: PHEME only ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("RUN A: Fine-tuning BERTweet on PHEME only")
    print("="*70)
    metrics_a = fine_tune(
        run_name  = "bertweet_pheme_only",
        X_train   = X_train_pheme,
        y_train   = y_train_pheme,
        X_test    = X_test,
        y_test    = y_test,
        tokenizer = tokenizer,
        save_dir  = "models/bertweet_pheme_only",
    )
    with open("metrics/bertweet_pheme_only.json", "w") as f:
        json.dump({**metrics_a, "dataset": "PHEME_only"}, f, indent=2)
    print(f"\nRun A metrics: {metrics_a}")

    # ── Pseudo-label live tweets using the PHEME-only model ───────────────────
    print("\n" + "="*70)
    print("PSEUDO-LABELING accumulated live tweets")
    print("="*70)
    pseudo_df = generate_pseudo_labels("models/bertweet_pheme_only")

    if len(pseudo_df) == 0:
        print("No confident pseudo-labels — skipping Run B.")
        comparison = {
            "pheme_only":  metrics_a,
            "augmented":   None,
            "delta":       None,
            "n_pseudo_labels": 0,
            "note": "Skipped: no confident pseudo-labels available",
        }
    else:
        # ── Build augmented training set ──────────────────────────────────────
        df_pheme_train = pd.DataFrame({"clean_title": X_train_pheme, "label": y_train_pheme})
        df_pheme_train['source']     = 'ground_truth'
        df_pheme_train['confidence'] = 1.0

        df_augmented = pd.concat([
            df_pheme_train[['clean_title', 'label', 'source', 'confidence']],
            pseudo_df[['clean_title',      'label', 'source', 'confidence']],
        ], ignore_index=True)

        os.makedirs('data/processed', exist_ok=True)
        df_augmented.to_csv('data/processed/pheme_augmented.csv', index=False)
        print(f"\nAugmented training set: {len(df_augmented)} examples "
              f"({len(X_train_pheme)} PHEME + {len(pseudo_df)} pseudo-labeled)")

        # ── Run B: PHEME + pseudo-labels ──────────────────────────────────────
        print("\n" + "="*70)
        print("RUN B: Fine-tuning BERTweet on PHEME + pseudo-labeled live tweets")
        print("="*70)
        metrics_b = fine_tune(
            run_name  = "bertweet_augmented",
            X_train   = df_augmented['clean_title'].tolist(),
            y_train   = df_augmented['label'].values,
            X_test    = X_test,
            y_test    = y_test,
            tokenizer = tokenizer,
            save_dir  = "models/bertweet_augmented",
            n_pseudo  = len(pseudo_df),
        )
        with open("metrics/bertweet_augmented.json", "w") as f:
            json.dump({**metrics_b, "dataset": "PHEME+PseudoLabels",
                       "n_pseudo": len(pseudo_df)}, f, indent=2)
        print(f"\nRun B metrics: {metrics_b}")

        # ── Comparison table ──────────────────────────────────────────────────
        delta = {k: round(metrics_b[k] - metrics_a[k], 4) for k in metrics_a}
        comparison = {
            "pheme_only":      metrics_a,
            "augmented":       metrics_b,
            "delta":           delta,
            "n_pseudo_labels": int(len(pseudo_df)),
            "n_pheme_train":   int(len(X_train_pheme)),
            "n_augmented_train": int(len(df_augmented)),
        }

        # ── Print side-by-side ────────────────────────────────────────────────
        print("\n" + "="*70)
        print(f"{'Metric':<15} {'PHEME-only':>12} {'PHEME+Pseudo':>14} {'Delta':>8}")
        print("-"*70)
        for k in metrics_a:
            a, b, d = metrics_a[k], metrics_b[k], delta[k]
            flag = " ▲" if d > 0 else (" ▼" if d < 0 else "")
            print(f"{k:<15} {a:>12.4f} {b:>14.4f} {d:>+8.4f}{flag}")
        print("="*70)

    with open("metrics/retraining_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("\nSaved: metrics/retraining_comparison.json")

    # Copy augmented model as the new production model
    if comparison.get("augmented") is not None:
        import shutil
        if os.path.exists("models/bertweet_finetuned"):
            shutil.rmtree("models/bertweet_finetuned")
        shutil.copytree("models/bertweet_augmented", "models/bertweet_finetuned")
        print("Production model updated to bertweet_augmented checkpoint.")


if __name__ == "__main__":
    main()
