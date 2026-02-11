import pandas as pd
import re
import string
import os

def clean_text(text):
    """Modular cleaning function for both training and real-time detection."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove links
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip() # Remove extra spaces
    return text

def preprocess_pipeline(input_path, output_path):
    if not os.path.exists(input_path):
        return
    df = pd.read_csv(input_path)
    df['clean_title'] = df['title'].apply(clean_text)
    df_final = df[['clean_title', 'label']]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_pipeline('data/raw/gossipcop_combined.csv', 'data/processed/gossipcop_cleaned.csv')