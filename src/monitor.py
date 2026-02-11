import pandas as pd
import tensorflow as tf
import pickle
import os
import sys
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocessing import clean_text

# US-specific keywords for quality monitoring
US_POLITICAL_KEYWORDS = ['congress', 'white house', 'biden', 'trump', 'senate', 'election', 'democrat', 'republican', 'washington']

def run_monitoring():
    # 1. Load latest scraped data
    data_dir = 'data/new_scraped/'
    if not os.listdir(data_dir):
        sys.exit(0)
    latest_file = max([os.path.join(data_dir, f) for f in os.listdir(data_dir)], key=os.path.getctime)
    df = pd.read_csv(latest_file)
    
    # 2. Relevancy Check: Ensure data is US-centric
    df['is_relevant'] = df['text'].str.lower().apply(lambda x: any(kw in x for kw in US_POLITICAL_KEYWORDS))
    relevancy_rate = df['is_relevant'].mean()
    print(f"US Political Relevancy Rate: {relevancy_rate:.2%}")

    # 3. Model Inference
    model = tf.keras.models.load_model('models/cnn_v1.h5')
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    df['cleaned'] = df['text'].apply(clean_text)
    sequences = tokenizer.texts_to_sequences(df['cleaned'])
    padded = pad_sequences(sequences, maxlen=50)
    predictions = model.predict(padded)
    
    avg_conf = np.mean(predictions)
    conf_variance = np.std(predictions)

    # 4. Final Trigger Logic
    # Trigger retraining if:
    # A) Data is no longer relevant (Source Drift)
    # B) Model is guessing (Confidence Drift)
    if relevancy_rate < 0.40 or avg_conf < 0.65 or conf_variance > 0.45:
        print("RETRAINING TRIGGERED: US data drift or performance drop detected.")
        sys.exit(1)
    else:
        print("System Stable: Model performing well on current US news.")
        sys.exit(0)

if __name__ == "__main__":
    run_monitoring()