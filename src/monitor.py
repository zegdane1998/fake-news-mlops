import pandas as pd
import tensorflow as tf
import pickle
import os
import sys
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocessing import clean_text
from src.train import MAX_LEN

# US-specific keywords for quality monitoring
US_POLITICAL_KEYWORDS = [
    'congress', 'white house', 'biden', 'trump', 'senate', 'election',
    'democrat', 'republican', 'washington', 'president', 'governor',
    'legislation', 'vote', 'ballot', 'campaign', 'policy', 'white house',
    'supreme court', 'house of representatives', 'filibuster'
]

def run_monitoring():
    # 1. Load latest scraped data
    data_dir = 'data/new_scraped/'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print("No scraped data found — skipping monitoring.")
        sys.exit(0)
    latest_file = max(
        [os.path.join(data_dir, f) for f in csv_files],
        key=os.path.getmtime
    )
    df = pd.read_csv(latest_file)
    print(f"Monitoring on: {latest_file} ({len(df)} articles)")

    # 2. Relevancy Check: Ensure data is US-centric
    df['is_relevant'] = df['text'].str.lower().apply(
        lambda x: any(kw in str(x) for kw in US_POLITICAL_KEYWORDS)
    )
    relevancy_rate = df['is_relevant'].mean()
    print(f"US Political Relevancy Rate: {relevancy_rate:.2%}")

    # 3. Model Inference
    model = tf.keras.models.load_model('models/cnn_v1.h5')
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    df['cleaned'] = df['text'].apply(clean_text)
    sequences = tokenizer.texts_to_sequences(df['cleaned'])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    predictions = model.predict(padded, verbose=0)

    avg_conf = np.mean(np.abs(predictions - 0.5)) * 2   # distance from decision boundary [0,1]
    pred_std  = np.std(predictions)

    print(f"Avg decision confidence: {avg_conf:.4f} (threshold: 0.30)")
    print(f"Prediction std-dev:      {pred_std:.4f}")

    # 4. Trigger retraining if:
    #    A) Data relevancy drops  — source drift
    #    B) Model collapses to always predicting one class — confidence collapse
    if relevancy_rate < 0.30 or avg_conf < 0.30:
        print("WARNING: Drift detected — triggering retraining.")
        sys.exit(1)

    print("System Stable: Model performing well on current US news.")
    sys.exit(0)

if __name__ == "__main__":
    run_monitoring()