import os
import pickle
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, callbacks

# Shared constants — imported by evaluate.py, monitor.py, baselines.py
MAX_WORDS = 30000   # larger vocab covers more political terminology
MAX_LEN   = 100     # longer context window for headline + subheadline patterns


def build_textcnn(vocab_size, embed_dim, max_len):
    """
    TextCNN: parallel convolutions with kernel sizes 3, 4, 5 capture
    different n-gram patterns (phrase-level to clause-level).
    Kim et al. (2014) — proven architecture for sentence classification.
    """
    inp = layers.Input(shape=(max_len,))
    emb = layers.Embedding(vocab_size, embed_dim)(inp)

    branches = []
    for kernel_size in (3, 4, 5):
        conv = layers.Conv1D(128, kernel_size, activation='relu', padding='same')(emb)
        pool = layers.GlobalMaxPooling1D()(conv)
        branches.append(pool)

    merged   = layers.Concatenate()(branches)        # 384-dim
    dropped  = layers.Dropout(0.5)(merged)
    dense    = layers.Dense(128, activation='relu')(dropped)
    dropped2 = layers.Dropout(0.3)(dense)
    out      = layers.Dense(1, activation='sigmoid')(dropped2)

    return models.Model(inputs=inp, outputs=out)


def train():
    os.makedirs('models', exist_ok=True)
    mlflow.set_experiment("fake-news-textcnn")

    with mlflow.start_run(run_name="textcnn_training"):

        # ── Log hyperparameters ────────────────────────────────────────────────
        mlflow.log_params({
            "max_words":          MAX_WORDS,
            "max_len":            MAX_LEN,
            "embed_dim":          128,
            "filters":            128,
            "kernel_sizes":       "3,4,5",
            "dropout_merged":     0.5,
            "dropout_dense":      0.3,
            "dense_units":        128,
            "batch_size":         64,
            "learning_rate":      1e-3,
            "epochs_max":         15,
            "early_stop_patience": 3,
            "reduce_lr_patience": 2,
            "reduce_lr_factor":   0.5,
            "class_weight":       "balanced",
        })

        # 1. Load data
        df = pd.read_csv('data/processed/gossipcop_cleaned.csv').dropna(
            subset=['clean_title', 'label']
        )
        df['label'] = df['label'].astype(int)
        X, y = df['clean_title'].astype(str), df['label']

        # 2. Stratified split — preserves class balance
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        mlflow.log_params({
            "train_size": len(X_train),
            "val_size":   len(X_val),
        })

        # 3. Tokenizer — fit only on training data to avoid leakage
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
        tokenizer.fit_on_texts(X_train)
        with open('models/tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Tokenizer saved ({len(tokenizer.word_index)} unique tokens found).")

        def encode(texts):
            return pad_sequences(
                tokenizer.texts_to_sequences(texts),
                maxlen=MAX_LEN, padding='post', truncating='post'
            )

        X_train_pad = encode(X_train)
        X_val_pad   = encode(X_val)

        # 4. Class weights — compensate for class imbalance
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"Class weights: {class_weight_dict}")

        # 5. Build and compile TextCNN
        model = build_textcnn(vocab_size=MAX_WORDS, embed_dim=128, max_len=MAX_LEN)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        model.summary()

        # 6. Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1
        )

        # 7. Train — capture history for MLflow logging
        history = model.fit(
            X_train_pad, y_train,
            validation_data=(X_val_pad, y_val),
            epochs=15,
            batch_size=64,
            class_weight=class_weight_dict,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        # ── Log training results ───────────────────────────────────────────────
        mlflow.log_metrics({
            "val_loss_best":     round(min(history.history['val_loss']), 4),
            "val_accuracy_best": round(max(history.history['val_accuracy']), 4),
            "epochs_trained":    len(history.history['loss']),
        })

        # 8. Save model + log artifacts
        model.save('models/cnn_v1.h5')
        mlflow.log_artifact('models/cnn_v1.h5',       artifact_path="model")
        mlflow.log_artifact('models/tokenizer.pkl',    artifact_path="model")
        mlflow.keras.log_model(model, artifact_path="keras_model")

        print(f"Model saved to models/cnn_v1.h5")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    train()
