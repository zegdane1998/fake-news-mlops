import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.train import MAX_WORDS, MAX_LEN

def evaluate():
    # ── Load model & tokenizer ──────────────────────────────────────────────
    print("Loading model and tokenizer...")
    model     = tf.keras.models.load_model('models/cnn_v1.h5')
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # ── Load cleaned dataset ────────────────────────────────────────────────
    print("Loading dataset...")
    df = pd.read_csv('data/processed/gossipcop_cleaned.csv').dropna(subset=['clean_title', 'label'])
    df['label'] = df['label'].astype(int)

    # ── Same 80/20 split as training (random_state=42, stratified) ──────────
    _, X_test, _, y_test = train_test_split(
        df['clean_title'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )

    # ── Tokenize & pad ──────────────────────────────────────────────────────
    print("Tokenizing test set...")
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

    # ── Predict ─────────────────────────────────────────────────────────────
    print("Running predictions...")
    y_prob = model.predict(X_test_pad, batch_size=64, verbose=1)
    y_pred = (y_prob >= 0.5).astype(int).flatten()
    y_true = y_test.values

    # ── Classification report ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(
        y_true, y_pred,
        target_names=['FAKE', 'REAL'],
        digits=4
    )
    print(report)

    # ── Confusion matrix ────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    print("CONFUSION MATRIX")
    print("="*60)
    print(f"              Predicted FAKE    Predicted REAL")
    print(f"Actual FAKE        {cm[0][0]}              {cm[0][1]}")
    print(f"Actual REAL        {cm[1][0]}              {cm[1][1]}")
    print("="*60)

    # ── Parse scalar metrics ─────────────────────────────────────────────────
    report_dict = classification_report(
        y_true, y_pred,
        target_names=['FAKE', 'REAL'],
        output_dict=True
    )

    accuracy  = report_dict['accuracy']
    precision = report_dict['macro avg']['precision']
    recall    = report_dict['macro avg']['recall']
    f1        = report_dict['macro avg']['f1-score']
    fake_f1   = report_dict['FAKE']['f1-score']
    real_f1   = report_dict['REAL']['f1-score']
    auc_roc   = roc_auc_score(y_true, y_prob.flatten())

    print(f"\nOverall Accuracy : {accuracy:.4f}")
    print(f"Macro Precision  : {precision:.4f}")
    print(f"Macro Recall     : {recall:.4f}")
    print(f"Macro F1-Score   : {f1:.4f}")
    print(f"AUC-ROC          : {auc_roc:.4f}")
    print(f"FAKE F1-Score    : {fake_f1:.4f}")
    print(f"REAL F1-Score    : {real_f1:.4f}")

    # ── Save metrics for DVC tracking ────────────────────────────────────────
    os.makedirs('metrics', exist_ok=True)
    metrics = {
        "accuracy":          round(accuracy, 4),
        "precision_macro":   round(precision, 4),
        "recall_macro":      round(recall, 4),
        "f1_macro":          round(f1, 4),
        "auc_roc":           round(auc_roc, 4),
        "f1_fake":           round(fake_f1, 4),
        "f1_real":           round(real_f1, 4),
        "confusion_matrix": {
            "TN": int(cm[0][0]),
            "FP": int(cm[0][1]),
            "FN": int(cm[1][0]),
            "TP": int(cm[1][1])
        }
    }
    with open('metrics/scores.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nMetrics saved to metrics/scores.json")
    print("Evaluation complete.")

if __name__ == "__main__":
    evaluate()