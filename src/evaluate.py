import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # must be before pyplot import — safe for headless CI
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 100  # must match src/train.py

# ── Plot helpers ───────────────────────────────────────────────────────────────

def save_roc_curve(y_true, y_prob, auc_val, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color='#00d4ff',
            label=f'TextCNN (AUC = {auc_val:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — TextCNN Fake News Classifier', fontsize=13)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"ROC curve saved to {output_path}")


def save_confidence_histogram(y_true, y_prob, output_path):
    fake_probs = y_prob[y_true == 0]
    real_probs = y_prob[y_true == 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(fake_probs, bins=30, alpha=0.6, color='#ff4d6d', label='FAKE (true)')
    ax.hist(real_probs, bins=30, alpha=0.6, color='#39ff7e', label='REAL (true)')
    ax.axvline(0.5, color='white', linestyle='--', lw=1.5, label='Decision boundary (0.5)')
    ax.set_xlabel('Predicted Probability (P=Real)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Confidence Distribution by True Class', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_facecolor('#0d1420')
    fig.patch.set_facecolor('#080c14')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Confidence histogram saved to {output_path}")


def per_class_error_analysis(X_test, y_true, y_pred, y_prob, output_path):
    errors_df = pd.DataFrame({
        'text':       X_test.values,
        'true_label': y_true,
        'pred_label': y_pred,
        'prob':       y_prob,
        'confidence': np.abs(y_prob - 0.5) * 2,
    })
    errors_df = errors_df[errors_df['true_label'] != errors_df['pred_label']].copy()

    fp = errors_df[errors_df['true_label'] == 0]  # FAKE predicted as REAL
    fn = errors_df[errors_df['true_label'] == 1]  # REAL predicted as FAKE

    def top_examples(group, n=10):
        return group.nlargest(n, 'confidence')['text'].tolist()

    report = {
        "false_positives": {
            "description": "FAKE articles incorrectly predicted as REAL",
            "count": int(len(fp)),
            "mean_confidence": round(float(fp['confidence'].mean()), 4) if len(fp) else 0.0,
            "top_examples": top_examples(fp),
        },
        "false_negatives": {
            "description": "REAL articles incorrectly predicted as FAKE",
            "count": int(len(fn)),
            "mean_confidence": round(float(fn['confidence'].mean()), 4) if len(fn) else 0.0,
            "top_examples": top_examples(fn),
        },
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nError Analysis:")
    print(f"  False Positives (FAKE→REAL): {report['false_positives']['count']}")
    print(f"  False Negatives (REAL→FAKE): {report['false_negatives']['count']}")
    print(f"Error analysis saved to {output_path}")
    return report


# ── Main ───────────────────────────────────────────────────────────────────────

def evaluate():
    os.makedirs('metrics', exist_ok=True)

    # ── Load model & tokenizer ─────────────────────────────────────────────────
    print("Loading model and tokenizer...")
    model = tf.keras.models.load_model('models/cnn_v1.h5')
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # ── Load cleaned dataset ───────────────────────────────────────────────────
    print("Loading dataset...")
    df = pd.read_csv('data/processed/gossipcop_cleaned.csv').dropna(
        subset=['clean_title', 'label']
    )
    df['label'] = df['label'].astype(int)

    # ── Same 80/20 split as training (random_state=42, stratified) ────────────
    _, X_test, _, y_test = train_test_split(
        df['clean_title'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )

    # ── Tokenize & pad ─────────────────────────────────────────────────────────
    print("Tokenizing test set...")
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN,
                               padding='post', truncating='post')

    # ── Predict ────────────────────────────────────────────────────────────────
    print("Running predictions...")
    y_prob = model.predict(X_test_pad, batch_size=64, verbose=1)
    y_pred = (y_prob >= 0.5).astype(int).flatten()
    y_true = y_test.values

    # ── Classification report ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(
        y_true, y_pred,
        target_names=['FAKE', 'REAL'],
        digits=4
    )
    print(report)

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print(f"              Predicted FAKE    Predicted REAL")
    print(f"Actual FAKE        {cm[0][0]}              {cm[0][1]}")
    print(f"Actual REAL        {cm[1][0]}              {cm[1][1]}")
    print("=" * 60)

    # ── Scalar metrics ─────────────────────────────────────────────────────────
    report_dict = classification_report(
        y_true, y_pred,
        target_names=['FAKE', 'REAL'],
        output_dict=True,
        zero_division=0
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

    # ── Plots ──────────────────────────────────────────────────────────────────
    save_roc_curve(y_true, y_prob.flatten(), auc_roc, 'metrics/roc_curve.png')
    save_confidence_histogram(y_true, y_prob.flatten(), 'metrics/confidence_histogram.png')

    # ── Per-class error analysis ───────────────────────────────────────────────
    per_class_error_analysis(
        X_test, y_true, y_pred, y_prob.flatten(),
        'metrics/error_analysis.json'
    )

    # ── Save metrics ───────────────────────────────────────────────────────────
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
            "TP": int(cm[1][1]),
        },
        "roc_curve_path":             "metrics/roc_curve.png",
        "confidence_histogram_path":  "metrics/confidence_histogram.png",
        "error_analysis_path":        "metrics/error_analysis.json",
    }
    with open('metrics/scores.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nMetrics saved to metrics/scores.json")
    print("Evaluation complete.")


if __name__ == "__main__":
    evaluate()
