import os
import json
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_WORDS, MAX_LEN = 30000, 100  # must match src/train.py

# ── Helpers ────────────────────────────────────────────────────────────────────

def _evaluate_model(y_true, y_prob, y_pred):
    report = classification_report(
        y_true, y_pred,
        target_names=['FAKE', 'REAL'],
        output_dict=True,
        zero_division=0
    )
    return {
        "accuracy":        round(report['accuracy'], 4),
        "precision_macro": round(report['macro avg']['precision'], 4),
        "recall_macro":    round(report['macro avg']['recall'], 4),
        "f1_macro":        round(report['macro avg']['f1-score'], 4),
        "f1_fake":         round(report['FAKE']['f1-score'], 4),
        "f1_real":         round(report['REAL']['f1-score'], 4),
        "auc_roc":         round(roc_auc_score(y_true, y_prob), 4),
    }


def _build_lstm(vocab_size, embed_dim, max_len):
    """Single-layer LSTM for sequence classification."""
    inp  = layers.Input(shape=(max_len,))
    emb  = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inp)
    x    = layers.LSTM(128, dropout=0.3, recurrent_dropout=0.3)(emb)
    x    = layers.Dense(64, activation='relu')(x)
    x    = layers.Dropout(0.3)(x)
    out  = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs=inp, outputs=out)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_baselines():
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)

    # ── Load data (same split as train.py / evaluate.py) ──────────────────────
    df = pd.read_csv('data/processed/gossipcop_cleaned.csv').dropna(
        subset=['clean_title', 'label']
    )
    df['label'] = df['label'].astype(int)
    X, y = df['clean_title'].astype(str), df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    results = {}
    mlflow.set_experiment("fake-news-baselines")

    # ── 1. TF-IDF + Logistic Regression ───────────────────────────────────────
    print("\n[1/3] Training TF-IDF + Logistic Regression...")
    with mlflow.start_run(run_name="tfidf_logreg"):
        params_lr = {
            "model_type": "TF-IDF + LogisticRegression",
            "max_features": 30000,
            "ngram_range": "1,2",
            "sublinear_tf": True,
            "C": 1.0,
            "class_weight": "balanced",
        }
        mlflow.log_params(params_lr)

        pipe_lr = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=30000, ngram_range=(1, 2), sublinear_tf=True
            )),
            ('clf', LogisticRegression(
                max_iter=1000, C=1.0, class_weight='balanced', random_state=42
            )),
        ])
        pipe_lr.fit(X_train, y_train)
        y_prob_lr = pipe_lr.predict_proba(X_test)[:, 1]
        y_pred_lr = pipe_lr.predict(X_test)

        metrics_lr = _evaluate_model(y_test.values, y_prob_lr, y_pred_lr)
        mlflow.log_metrics(metrics_lr)
        mlflow.sklearn.log_model(pipe_lr, artifact_path="model")

        pickle.dump(pipe_lr, open('models/tfidf_logreg.pkl', 'wb'))
        mlflow.log_artifact('models/tfidf_logreg.pkl')

        results['TF-IDF + LogReg'] = metrics_lr
        print(f"  AUC-ROC: {metrics_lr['auc_roc']}  F1-macro: {metrics_lr['f1_macro']}")

    # ── 2. TF-IDF + SVM (calibrated for probabilities) ────────────────────────
    print("\n[2/3] Training TF-IDF + SVM...")
    with mlflow.start_run(run_name="tfidf_svm"):
        params_svm = {
            "model_type": "TF-IDF + SVM (calibrated)",
            "max_features": 30000,
            "ngram_range": "1,2",
            "sublinear_tf": True,
            "C": 1.0,
            "class_weight": "balanced",
            "calibration": "sigmoid",
        }
        mlflow.log_params(params_svm)

        pipe_svm = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=30000, ngram_range=(1, 2), sublinear_tf=True
            )),
            ('clf', CalibratedClassifierCV(
                LinearSVC(max_iter=2000, C=1.0, class_weight='balanced'),
                cv=3, method='sigmoid'
            )),
        ])
        pipe_svm.fit(X_train, y_train)
        y_prob_svm = pipe_svm.predict_proba(X_test)[:, 1]
        y_pred_svm = pipe_svm.predict(X_test)

        metrics_svm = _evaluate_model(y_test.values, y_prob_svm, y_pred_svm)
        mlflow.log_metrics(metrics_svm)
        mlflow.sklearn.log_model(pipe_svm, artifact_path="model")

        pickle.dump(pipe_svm, open('models/tfidf_svm.pkl', 'wb'))
        mlflow.log_artifact('models/tfidf_svm.pkl')

        results['TF-IDF + SVM'] = metrics_svm
        print(f"  AUC-ROC: {metrics_svm['auc_roc']}  F1-macro: {metrics_svm['f1_macro']}")

    # ── 3. LSTM (reuses tokenizer from train.py to share vocabulary) ──────────
    print("\n[3/3] Training LSTM...")
    with mlflow.start_run(run_name="lstm"):
        params_lstm = {
            "model_type": "LSTM",
            "vocab_size": MAX_WORDS,
            "embed_dim": 128,
            "lstm_units": 128,
            "lstm_dropout": 0.3,
            "dense_units": 64,
            "dense_dropout": 0.3,
            "max_len": MAX_LEN,
            "batch_size": 64,
            "early_stop_patience": 3,
        }
        mlflow.log_params(params_lstm)

        with open('models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)

        def encode(texts):
            return pad_sequences(
                tokenizer.texts_to_sequences(texts),
                maxlen=MAX_LEN, padding='post', truncating='post'
            )

        X_tr_pad = encode(X_train)
        X_te_pad = encode(X_test)

        lstm_model = _build_lstm(MAX_WORDS, embed_dim=128, max_len=MAX_LEN)
        lstm_model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
        )

        history = lstm_model.fit(
            X_tr_pad, y_train,
            validation_split=0.1,
            epochs=15,
            batch_size=64,
            callbacks=[early_stop],
            verbose=1
        )

        mlflow.log_metrics({
            "val_loss_best": round(min(history.history['val_loss']), 4),
            "val_accuracy_best": round(max(history.history['val_accuracy']), 4),
            "epochs_trained": len(history.history['loss']),
        })

        y_prob_lstm = lstm_model.predict(X_te_pad, verbose=0).flatten()
        y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)

        metrics_lstm = _evaluate_model(y_test.values, y_prob_lstm, y_pred_lstm)
        mlflow.log_metrics(metrics_lstm)
        mlflow.keras.log_model(lstm_model, artifact_path="keras_model")

        lstm_model.save('models/lstm_v1.h5')
        mlflow.log_artifact('models/lstm_v1.h5')

        results['LSTM'] = metrics_lstm
        print(f"  AUC-ROC: {metrics_lstm['auc_roc']}  F1-macro: {metrics_lstm['f1_macro']}")

    # ── 4. Include TextCNN results for complete comparison table ───────────────
    if os.path.exists('metrics/scores.json'):
        with open('metrics/scores.json') as f:
            cnn_scores = json.load(f)
        results['TextCNN'] = {k: v for k, v in cnn_scores.items()
                              if k != 'confusion_matrix'}
        with mlflow.start_run(run_name="textcnn_reference"):
            mlflow.log_param("model_type", "TextCNN (pre-trained reference)")
            mlflow.log_metrics({k: v for k, v in results['TextCNN'].items()
                                if isinstance(v, (int, float))})

    # ── Save comparison table ──────────────────────────────────────────────────
    with open('metrics/baselines.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':>9} {'F1-macro':>9} {'AUC-ROC':>9}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<25} {m['accuracy']:>9.4f} {m['f1_macro']:>9.4f} {m['auc_roc']:>9.4f}")
    print("=" * 60)
    print("\nBaseline results saved to metrics/baselines.json")


if __name__ == "__main__":
    run_baselines()
