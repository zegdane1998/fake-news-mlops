import argparse
import os
import re
import string

import pandas as pd


def clean_text(text):
    """Standard cleaning for news headlines (existing pipeline)."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_tweet(text):
    """
    Minimal cleaning for BERTweet — keep URLs/mentions as normalised tokens
    (download_pheme.py already replaced them with HTTPURL / @USER).
    Only strip extra whitespace.
    """
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_pipeline(input_path, output_path, mode="headline"):
    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    if mode == "tweet":
        df['clean_title'] = df['text'].apply(clean_tweet)
    else:
        df['clean_title'] = df['title'].apply(clean_text)

    df_final = df[['clean_title', 'label']]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Preprocessed {len(df_final)} rows → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw/gossipcop_combined.csv")
    parser.add_argument("--output", default="data/processed/gossipcop_cleaned.csv")
    parser.add_argument("--mode",   default="headline", choices=["headline", "tweet"])
    args = parser.parse_args()
    preprocess_pipeline(args.input, args.output, args.mode)
