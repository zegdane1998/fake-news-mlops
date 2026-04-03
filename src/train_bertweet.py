"""
Fine-tune BERTweet (vinai/bertweet-base) on the PHEME tweet dataset.
Tracks everything with MLflow. Saves the model to models/bertweet_finetuned/.
"""

import json
import os

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ── Load params ────────────────────────────────────────────────────────────────
with open("params.yaml") as f:
    P = yaml.safe_load(f)["bertweet"]

MODEL_NAME   = P["model_name"]
MAX_LEN      = P["max_len"]
BATCH_SIZE   = P["batch_size"]
EPOCHS       = P["epochs"]
LR           = P["learning_rate"]
WARMUP_RATIO = P["warmup_ratio"]
WEIGHT_DECAY = P["weight_decay"]
TEST_SIZE    = P["test_size"]
RANDOM_STATE = P["random_state"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ── Dataset ────────────────────────────────────────────────────────────────────
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(
            list(texts),
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
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


# ── Training loop ──────────────────────────────────────────────────────────────
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


def eval_epoch(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )
            total_loss += out.loss.item()
            probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())
    return (
        total_loss / len(loader),
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_labels),
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def train():
    os.makedirs("models/bertweet_finetuned", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    mlflow.set_experiment("fake-news-bertweet")

    with mlflow.start_run(run_name="bertweet_pheme"):
        mlflow.log_params(P)

        # 1. Data
        df = pd.read_csv("data/processed/pheme_cleaned.csv").dropna(
            subset=["clean_title", "label"]
        )
        df["label"] = df["label"].astype(int)
        X, y = df["clean_title"].astype(str), df["label"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        mlflow.log_params({"train_size": len(X_train), "val_size": len(X_val)})
        print(f"Train: {len(X_train)}  Val: {len(X_val)}")

        # 2. Tokenizer + datasets
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        train_ds = TweetDataset(X_train, y_train, tokenizer)
        val_ds   = TweetDataset(X_val,   y_val,   tokenizer)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        # 3. Model
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        ).to(DEVICE)

        # 4. Optimiser + scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        total_steps  = len(train_loader) * EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # 5. Training
        best_f1 = 0.0
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler)
            val_loss, preds, probs, labels = eval_epoch(model, val_loader)

            acc  = accuracy_score(labels, preds)
            f1   = f1_score(labels, preds, average="macro")
            auc  = roc_auc_score(labels, probs)

            print(
                f"Epoch {epoch}/{EPOCHS} — "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}"
            )
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss,
                 "accuracy": acc, "f1_macro": f1, "roc_auc": auc},
                step=epoch,
            )

            if f1 > best_f1:
                best_f1 = f1
                model.save_pretrained("models/bertweet_finetuned")
                tokenizer.save_pretrained("models/bertweet_finetuned")
                print(f"  ✓ Best model saved (f1={best_f1:.4f})")

        # 6. Final evaluation on best model
        model = AutoModelForSequenceClassification.from_pretrained(
            "models/bertweet_finetuned"
        ).to(DEVICE)
        _, preds, probs, labels = eval_epoch(model, val_loader)

        report = classification_report(labels, preds, target_names=["fake", "real"])
        print("\nClassification Report:\n", report)

        scores = {
            "accuracy":  round(float(accuracy_score(labels, preds)), 4),
            "f1_macro":  round(float(f1_score(labels, preds, average="macro")), 4),
            "f1_fake":   round(float(f1_score(labels, preds, pos_label=0)), 4),
            "f1_real":   round(float(f1_score(labels, preds, pos_label=1)), 4),
            "roc_auc":   round(float(roc_auc_score(labels, probs)), 4),
        }
        with open("metrics/bertweet_scores.json", "w") as f:
            json.dump(scores, f, indent=2)

        mlflow.log_metrics(scores)
        mlflow.log_artifact("metrics/bertweet_scores.json")
        mlflow.log_artifact("models/bertweet_finetuned", artifact_path="model")

        print(f"\nFinal scores: {scores}")


if __name__ == "__main__":
    train()
