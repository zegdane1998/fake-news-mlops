#!/bin/bash
# GPU training script for Vast.ai
# Runs BERTweet fine-tuning on PHEME, then pushes results to GitHub.
#
# Usage:
#   bash train_vastai.sh <GITHUB_TOKEN>

set -e

GITHUB_TOKEN="$1"
REPO="zegdane1998/fake-news-mlops"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "ERROR: provide your GitHub token"
    exit 1
fi

push_status() {
    local msg="$1 [skip ci]"
    git pull origin master --rebase --quiet 2>/dev/null || true
    git add -A 2>/dev/null || true
    git diff --cached --quiet && \
        git commit --allow-empty -m "$msg" || \
        git commit -m "$msg"
    git push origin master
}

echo "=== [1/5] System setup ==="
apt-get update -qq && apt-get install -y -qq git git-lfs
git lfs install

echo "=== [2/5] Clone repo ==="
git config --global http.postBuffer 524288000
git clone --depth 1 https://${GITHUB_TOKEN}@github.com/${REPO}.git
cd fake-news-mlops

git config user.email "24COMP5001@isik.edu.tr"
git config user.name "Abdellah Zegdane"
git remote set-url origin https://${GITHUB_TOKEN}@github.com/${REPO}.git

# Push failure status if anything below exits unexpectedly
trap 'LAST_ERR=$(tail -10 /root/train.log 2>/dev/null | tr "\n" " " | cut -c1-300); push_status "Vast.ai: training FAILED at $(date -u +%H:%M:%S) — $LAST_ERR"' ERR

push_status "Vast.ai: training started $(date -u '+%Y-%m-%d %H:%M')"

echo "=== [3/5] Install Python dependencies ==="
# Ensure conda environment is active so CUDA libs are on PATH
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true
pip install --quiet --upgrade pip
pip install --quiet \
    transformers==4.40.0 \
    datasets \
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

push_status "Vast.ai: data ready, starting BERTweet fine-tuning $(date -u '+%H:%M')"

echo "--- Fine-tuning BERTweet ---"
python src/train_bertweet.py

echo "=== [5/5] Push results to GitHub ==="
# fetch + reset avoids rebase conflicts if master moved while training ran
git fetch origin master
git reset --mixed origin/master
git add metrics/bertweet_scores.json metrics/baselines.json
git diff --cached --quiet || git commit -m "Vast.ai: BERTweet fine-tuned on PHEME $(date -u '+%Y-%m-%d')"
git push origin master
trap - ERR

echo ""
echo "============================================"
echo "  DONE! Results pushed to GitHub."
cat metrics/bertweet_scores.json
echo "============================================"
