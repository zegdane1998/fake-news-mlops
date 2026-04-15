#!/bin/bash
# GPU training script for Vast.ai
# Runs BERTweet fine-tuning on PHEME, then pushes results to GitHub.
#
# Usage:
#   bash train_vastai.sh <GITHUB_TOKEN>
#
# Get a token at: GitHub → Settings → Developer settings → Personal access tokens → Fine-grained
# Required permissions: Contents (read & write) on zegdane1998/fake-news-mlops

set -e

GITHUB_TOKEN="$1"
REPO="zegdane1998/fake-news-mlops"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "ERROR: provide your GitHub token"
    echo "Usage: bash train_vastai.sh <GITHUB_TOKEN>"
    exit 1
fi

echo "=== [1/5] System setup ==="
apt-get update -qq && apt-get install -y -qq git

echo "=== [2/5] Clone repo ==="
git config --global http.postBuffer 524288000
git clone --depth 1 https://${GITHUB_TOKEN}@github.com/${REPO}.git
cd fake-news-mlops

git config user.email "24COMP5001@isik.edu.tr"
git config user.name "Abdellah Zegdane"

echo "=== [3/5] Install Python dependencies ==="
pip install --quiet --upgrade pip
pip install --quiet \
    transformers==4.40.0 \
    datasets \
    torch \
    mlflow \
    pandas numpy scikit-learn \
    pyyaml requests tqdm accelerate

echo "=== [4/5] Run training pipeline ==="
echo "--- Downloading PHEME from figshare ---"
python src/download_pheme.py

echo "--- Preprocessing tweets ---"
python src/preprocessing.py \
    --input  data/raw/pheme_tweets.csv \
    --output data/processed/pheme_cleaned.csv \
    --mode   tweet

echo "--- Fine-tuning BERTweet ---"
python src/train_bertweet.py

echo "=== [5/5] Push results to GitHub ==="
git pull origin master --rebase
git add metrics/bertweet_scores.json metrics/baselines.json
git commit -m "Vast.ai: BERTweet fine-tuned on PHEME $(date -u '+%Y-%m-%d')"
git push origin master

echo ""
echo "============================================"
echo "  DONE! Results pushed to GitHub."
cat metrics/bertweet_scores.json
echo "============================================"
