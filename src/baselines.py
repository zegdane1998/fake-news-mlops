"""
Train and compare all baseline models on PHEME tweet data.
Models: TF-IDF+LogReg, TF-IDF+SVM, LSTM, TextCNN
All trained on the same PHEME split as BERTweet for a fair comparison.
Results saved to metrics/pheme_baselines.json
"""

import json
import os
import pickle

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

DATA_PATH  = "data/processed/pheme_cleaned.csv"
VOCAB_SIZE = 20000
MAX_LEN    = 128
EMBED_DIM  = 128
BATCH_SIZE = 64
EPOCHS     = 15
PATIENCE   = 3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42


# ── Vocabulary & encoding ──────────────────────────────────────────────────────

class Vocabulary:
    def __init__(self, max_words=VOCAB_SIZE):
        self.max_words = max_words
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}

    def build(self, texts):
        from collections import Counter
        counts = Counter(w for t in texts for w in t.lower().split())
        for word, _ in counts.most_common(self.max_words - 2):
            self.word2idx[word] = len(self.word2idx)

    def encode(self, texts, max_len=MAX_LEN):
        out = []
        for text in texts:
            ids = [self.word2idx.get(w, 1) for w in text.lower().split()]
            ids = ids[:max_len]
            ids += [0] * (max_len - len(ids))
            out.append(ids)
        return np.array(out, dtype=np.int64)


# ── PyTorch models ─────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden=128):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)  # single layer, no dropout
        self.fc   = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x):
        _, (h, _) = self.lstm(self.emb(x))
        return self.fc(h.squeeze(0)).squeeze(-1)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in (3, 4, 5)
        ])
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_filters * 3, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 1)
        )

    def forward(self, x):
        e = self.emb(x).permute(0, 2, 1)
        pooled = [torch.relu(c(e)).max(dim=-1).values for c in self.convs]
        return self.fc(torch.cat(pooled, dim=1)).squeeze(-1)


# ── Training loop ──────────────────────────────────────────────────────────────

def _oversample(X, y):
    """Oversample minority class to balance training data."""
    fake_idx = np.where(y == 0)[0]
    real_idx = np.where(y == 1)[0]
    if len(fake_idx) >= len(real_idx):
        return X, y
    repeats = len(real_idx) // len(fake_idx)
    remainder = len(real_idx) % len(fake_idx)
    fake_idx_bal = np.concatenate([
        np.tile(fake_idx, repeats),
        np.random.RandomState(42).choice(fake_idx, remainder, replace=False)
    ])
    idx = np.concatenate([real_idx, fake_idx_bal])
    np.random.RandomState(42).shuffle(idx)
    return X[idx], y[idx]


def _train_pytorch(model, X_tr, y_tr, X_val, y_val):
    X_tr_bal, y_tr_bal = _oversample(X_tr, y_tr)
    print(f"  Balanced train: {(y_tr_bal==0).sum()} fake / {(y_tr_bal==1).sum()} real")

    tr_ds  = TensorDataset(torch.tensor(X_tr_bal), torch.tensor(y_tr_bal, dtype=torch.float))
    val_ds = TensorDataset(torch.tensor(X_val),    torch.tensor(y_val,    dtype=torch.float))
    tr_loader  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val, patience_cnt, best_state = float("inf"), 0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

        print(f"  Epoch {epoch}/{EPOCHS} — val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model


def _predict_pytorch(model, X):
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X)), batch_size=256)
    probs = []
    with torch.no_grad():
        for (xb,) in loader:
            probs.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().tolist())
    return np.array(probs)


# ── Metrics ────────────────────────────────────────────────────────────────────

def _metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro":  round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "f1_fake":   round(float(f1_score(y_true, y_pred, pos_label=0)), 4),
        "f1_real":   round(float(f1_score(y_true, y_pred, pos_label=1)), 4),
        "auc_roc":   round(float(roc_auc_score(y_true, y_prob)), 4),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run_baselines():
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    print(f"Device: {DEVICE}")

    df = pd.read_csv(DATA_PATH).dropna(subset=["clean_title", "label"])
    df["label"] = df["label"].astype(int)
    X, y = df["clean_title"].astype(str).tolist(), df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"  Fake (0): {(y_test==0).sum()}  Real (1): {(y_test==1).sum()}")

    results = {}
    mlflow.set_experiment("pheme-baselines")

    # ── 1. TF-IDF + Logistic Regression ───────────────────────────────────────
    print("\n[1/4] TF-IDF + Logistic Regression")
    with mlflow.start_run(run_name="tfidf_logreg"):
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf",   LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=RANDOM_STATE)),
        ])
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        m = _metrics(y_test, probs)
        mlflow.log_metrics(m)
        pickle.dump(pipe, open("models/tfidf_logreg.pkl", "wb"))
        results["TF-IDF + LogReg"] = m
        print(f"  AUC={m['auc_roc']}  F1={m['f1_macro']}")

    # ── 2. TF-IDF + SVM ───────────────────────────────────────────────────────
    print("\n[2/4] TF-IDF + SVM")
    with mlflow.start_run(run_name="tfidf_svm"):
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf",   CalibratedClassifierCV(
                LinearSVC(max_iter=2000, C=1.0, class_weight="balanced"), cv=3, method="sigmoid"
            )),
        ])
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        m = _metrics(y_test, probs)
        mlflow.log_metrics(m)
        pickle.dump(pipe, open("models/tfidf_svm.pkl", "wb"))
        results["TF-IDF + SVM"] = m
        print(f"  AUC={m['auc_roc']}  F1={m['f1_macro']}")

    # ── 3. LSTM ────────────────────────────────────────────────────────────────
    print("\n[3/4] LSTM")
    vocab = Vocabulary()
    vocab.build(X_train)
    X_tr_enc = vocab.encode(X_train)
    X_te_enc = vocab.encode(X_test)
    X_tr_val, X_val_enc, y_tr_val, y_val_enc = train_test_split(
        X_tr_enc, y_train, test_size=0.1, random_state=RANDOM_STATE
    )

    with mlflow.start_run(run_name="lstm"):
        model = LSTMClassifier(VOCAB_SIZE, EMBED_DIM).to(DEVICE)
        model = _train_pytorch(model, X_tr_val, y_tr_val, X_val_enc, y_val_enc)
        probs = _predict_pytorch(model, X_te_enc)
        m = _metrics(y_test, probs)
        mlflow.log_metrics(m)
        torch.save(model.state_dict(), "models/lstm_pheme.pt")
        pickle.dump(vocab, open("models/vocab_pheme.pkl", "wb"))
        results["LSTM"] = m
        print(f"  AUC={m['auc_roc']}  F1={m['f1_macro']}")

    # ── 4. TextCNN ─────────────────────────────────────────────────────────────
    print("\n[4/4] TextCNN")
    with mlflow.start_run(run_name="textcnn"):
        model = TextCNN(VOCAB_SIZE, EMBED_DIM).to(DEVICE)
        model = _train_pytorch(model, X_tr_val, y_tr_val, X_val_enc, y_val_enc)
        probs = _predict_pytorch(model, X_te_enc)
        m = _metrics(y_test, probs)
        mlflow.log_metrics(m)
        torch.save(model.state_dict(), "models/textcnn_pheme.pt")
        results["TextCNN"] = m
        print(f"  AUC={m['auc_roc']}  F1={m['f1_macro']}")

    # ── Merge BERTweet results ─────────────────────────────────────────────────
    if os.path.exists("metrics/bertweet_scores.json"):
        with open("metrics/bertweet_scores.json") as f:
            bt = json.load(f)
        # normalize key name (train_bertweet.py saves as 'roc_auc')
        if "roc_auc" in bt and "auc_roc" not in bt:
            bt["auc_roc"] = bt.pop("roc_auc")
        results["BERTweet"] = bt

    # ── Save & print table ─────────────────────────────────────────────────────
    with open("metrics/pheme_baselines.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print(f"{'Model':<22} {'Acc':>7} {'F1-macro':>9} {'F1-fake':>8} {'AUC':>7}")
    print("-" * 65)
    for name, m in results.items():
        print(f"{name:<22} {m['accuracy']:>7.4f} {m['f1_macro']:>9.4f} {m['f1_fake']:>8.4f} {m['auc_roc']:>7.4f}")
    print("=" * 65)
    print("Saved to metrics/pheme_baselines.json")


if __name__ == "__main__":
    run_baselines()
