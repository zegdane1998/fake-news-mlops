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
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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


# ── Shared test-split helpers (saved to disk so steps A and B use same split) ──

SPLIT_PATH = "data/processed/test_split.pkl"

def _load_pheme():
    df = pd.read_csv('data/processed/pheme_cleaned.csv').dropna(
        subset=['clean_title', 'label']
    )
    df['label'] = df['label'].astype(int)
    X_all = df['clean_title'].astype(str).tolist()
    y_all = df['label'].values
    return train_test_split(X_all, y_all, test_size=0.2,
                            random_state=RANDOM_STATE, stratify=y_all)

def _save_split(X_train, X_test, y_train, y_test):
    import pickle
    os.makedirs('data/processed', exist_ok=True)
    with open(SPLIT_PATH, 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)

def _load_split():
    import pickle
    with open(SPLIT_PATH, 'rb') as f:
        return pickle.load(f)


# ── Step functions ────────────────────────────────────────────────────────────

def step_a():
    """Train BERTweet on PHEME only. Saves model + test split for step_b."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    X_train, X_test, y_train, y_test = _load_pheme()
    _save_split(X_train, X_test, y_train, y_test)
    print(f"Test set: {len(X_test)} tweets  "
          f"({(y_test==0).sum()} fake / {(y_test==1).sum()} real)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    print("\n" + "="*70)
    print("RUN A: Fine-tuning BERTweet on PHEME only")
    print("="*70)
    m = fine_tune("bertweet_pheme_only", X_train, y_train,
                  X_test, y_test, tokenizer, "models/bertweet_pheme_only")
    with open("metrics/bertweet_pheme_only.json", "w") as f:
        json.dump({**m, "dataset": "PHEME_only"}, f, indent=2)
    print(f"\nRun A done: {m}")


def step_pseudo():
    """Pseudo-label live tweets using the PHEME-only model."""
    print("\n" + "="*70)
    print("PSEUDO-LABELING accumulated live tweets")
    print("="*70)
    X_train, X_test, y_train, y_test = _load_split()
    pseudo_df = generate_pseudo_labels("models/bertweet_pheme_only")
    if len(pseudo_df) == 0:
        print("No confident pseudo-labels — Run B will use PHEME only.")
        return
    df_pheme_train = pd.DataFrame({"clean_title": X_train, "label": y_train,
                                    "source": "ground_truth", "confidence": 1.0})
    df_aug = pd.concat([
        df_pheme_train[['clean_title', 'label', 'source', 'confidence']],
        pseudo_df[['clean_title', 'label', 'source', 'confidence']],
    ], ignore_index=True)
    os.makedirs('data/processed', exist_ok=True)
    df_aug.to_csv('data/processed/pheme_augmented.csv', index=False)
    print(f"\nAugmented set: {len(df_aug)} examples "
          f"({len(X_train)} PHEME + {len(pseudo_df)} pseudo-labeled)")


def step_b():
    """Train BERTweet on PHEME + pseudo-labels."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    _, X_test, _, y_test = _load_split()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    aug_path = 'data/processed/pheme_augmented.csv'
    if os.path.exists(aug_path):
        df_aug = pd.read_csv(aug_path).dropna(subset=['clean_title', 'label'])
        df_aug['label'] = df_aug['label'].astype(int)
        X_train = df_aug['clean_title'].tolist()
        y_train = df_aug['label'].values
        n_pseudo = int((df_aug['source'] == 'pseudo_label').sum())
    else:
        print("No augmented dataset — falling back to PHEME only for Run B.")
        X_train, _, y_train, _ = _load_split()
        n_pseudo = 0

    print("\n" + "="*70)
    print("RUN B: Fine-tuning BERTweet on PHEME + pseudo-labeled live tweets")
    print("="*70)
    m = fine_tune("bertweet_augmented", X_train, y_train,
                  X_test, y_test, tokenizer, "models/bertweet_augmented",
                  n_pseudo=n_pseudo)
    with open("metrics/bertweet_augmented.json", "w") as f:
        json.dump({**m, "dataset": "PHEME+PseudoLabels", "n_pseudo": n_pseudo}, f, indent=2)
    print(f"\nRun B done: {m}")

    # Promote to production model
    import shutil
    if os.path.exists("models/bertweet_finetuned"):
        shutil.rmtree("models/bertweet_finetuned")
    shutil.copytree("models/bertweet_augmented", "models/bertweet_finetuned")
    print("Production model updated to bertweet_augmented.")


def step_compare():
    """Generate side-by-side comparison table from saved metrics."""
    with open("metrics/bertweet_pheme_only.json") as f:
        ma = json.load(f)
    with open("metrics/bertweet_augmented.json") as f:
        mb = json.load(f)

    keys  = ["accuracy", "f1_macro", "f1_fake", "f1_real", "auc_roc"]
    ma    = {k: ma[k] for k in keys}
    mb    = {k: mb[k] for k in keys}
    delta = {k: round(mb[k] - ma[k], 4) for k in keys}

    _, X_test, _, y_test = _load_split()
    aug_path = 'data/processed/pheme_augmented.csv'
    n_pseudo = 0
    if os.path.exists(aug_path):
        df_aug = pd.read_csv(aug_path)
        n_pseudo = int((df_aug['source'] == 'pseudo_label').sum()) if 'source' in df_aug.columns else 0

    comparison = {
        "pheme_only":        ma,
        "augmented":         mb,
        "delta":             delta,
        "n_pseudo_labels":   n_pseudo,
        "n_test":            len(X_test),
    }
    with open("metrics/retraining_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "="*70)
    print(f"{'Metric':<15} {'PHEME-only':>12} {'PHEME+Pseudo':>14} {'Delta':>8}")
    print("-"*70)
    for k in keys:
        flag = " ▲" if delta[k] > 0 else (" ▼" if delta[k] < 0 else "")
        print(f"{k:<15} {ma[k]:>12.4f} {mb[k]:>14.4f} {delta[k]:>+8.4f}{flag}")
    print("="*70)
    print("\nSaved: metrics/retraining_comparison.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step",
                        choices=["a", "pseudo", "b", "compare"],
                        required=True,
                        help="Which step to run")
    args = parser.parse_args()

    if   args.step == "a":       step_a()
    elif args.step == "pseudo":  step_pseudo()
    elif args.step == "b":       step_b()
    elif args.step == "compare": step_compare()
