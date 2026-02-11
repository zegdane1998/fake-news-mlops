import pandas as pd
import tensorflow as tf
import pickle
import os
import sys
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocessing import clean_text

def run_monitoring():
    # 1. Locate the latest scraped file
    data_dir = 'data/new_scraped/'
    if not os.listdir(data_dir):
        print("Monitoring skipped: No new data found.")
        sys.exit(0)
        
    latest_file = max([os.path.join(data_dir, f) for f in os.listdir(data_dir)], key=os.path.getctime)
    df = pd.read_csv(latest_file)
    
    # 2. Load Model Artifacts
    model = tf.keras.models.load_model('models/cnn_v1.h5')
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # 3. Preprocess and Run Inference
    df['cleaned'] = df['text'].apply(clean_text)
    sequences = tokenizer.texts_to_sequences(df['cleaned'])
    padded = pad_sequences(sequences, maxlen=50)
    
    predictions = model.predict(padded)
    
    # 4. MLOps Trigger Logic: Drift and Confidence
    # We use Standard Deviation as a proxy for "Model Conflict"
    avg_conf = np.mean(predictions)
    conf_variance = np.std(predictions)

    print(f"Monitoring Report for {os.path.basename(latest_file)}:")
    print(f" - Average Confidence: {avg_conf:.4f}")
    print(f" - Confidence Variance: {conf_variance:.4f}")

    # RETRAINING TRIGGER: If confidence is too low or variance is too high
    # This indicates the model is guessing or seeing data it doesn't recognize (Drift)
    if avg_conf < 0.65 or conf_variance > 0.40:
        print("ALERT: Performance drop or data drift detected!")
        sys.exit(1) # Exit with error code 1 to trigger 'python src/train.py' in GitHub Actions
    else:
        print("Performance within acceptable range. No retraining needed.")
        sys.exit(0)

if __name__ == "__main__":
    run_monitoring()