#!/bin/bash
# GPU training script for Vast.ai
#
# Full pipeline:
#   1. Prepare data (PHEME download + preprocessing)
#   2. Run comparison experiment:
#        a. Fine-tune BERTweet on PHEME only   → metrics/bertweet_pheme_only.json
#        b. Pseudo-label 1,100+ accumulated live tweets using that model
#        c. Fine-tune BERTweet on PHEME+pseudo → metrics/bertweet_augmented.json
#        d. Side-by-side comparison             → metrics/retraining_comparison.json
#   3. Run drift analysis (mixing experiment)   → metrics/drift_analysis.json
#   4. Rebuild reference distribution for monitor using the new production model
#   5. Push all results to GitHub
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
    local msg="$1"
    git pull origin master --rebase --quiet 2>/dev/null || true
    # Only add small files — never model checkpoints (500MB+ would hang git push)
    git add metrics/*.json data/processed/*.csv data/processed/*.pkl \
        2>/dev/null || true
    git diff --cached --quiet && \
        git commit --allow-empty -m "$msg" || \
        git commit -m "$msg"
    git push origin master
}

echo "=== [1/6] System setup ==="
apt-get update -qq && apt-get install -y -qq git git-lfs
git lfs install

echo "=== [2/6] Clone repo ==="
git config --global http.postBuffer 524288000
git clone --depth 1 https://${GITHUB_TOKEN}@github.com/${REPO}.git
cd fake-news-mlops

git config user.email "24COMP5001@isik.edu.tr"
git config user.name "Abdellah Zegdane"
git remote set-url origin https://${GITHUB_TOKEN}@github.com/${REPO}.git

trap 'LAST_ERR=$(tail -10 /root/train.log 2>/dev/null | tr "\n" " " | cut -c1-300); push_status "Vast.ai: FAILED at $(date -u +%H:%M:%S) — $LAST_ERR"' ERR

push_status "Vast.ai: training started $(date -u '+%Y-%m-%d %H:%M')"

echo "=== [3/6] Install Python dependencies ==="
# pytorch/pytorch image has Python + PyTorch pre-installed via conda
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true
pip install --quiet \
    "transformers==4.40.0" \
    datasets \
    pandas numpy "scikit-learn>=1.3" \
    pyyaml requests tqdm accelerate \
    scipy sentencepiece emoji

echo "=== [4/6] Prepare PHEME data ==="
echo "--- Downloading PHEME from figshare ---"
python src/download_pheme.py

echo "--- Preprocessing tweets ---"
python src/preprocessing.py \
    --input  data/raw/pheme_tweets.csv \
    --output data/processed/pheme_cleaned.csv \
    --mode   tweet

push_status "Vast.ai: data ready, starting comparison experiment $(date -u '+%H:%M')"

echo "=== [5/6] Run comparison experiment ==="

echo "--- Pre-downloading BERTweet model from HuggingFace (~500 MB) ---"
python - <<'PYEOF'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml, os
with open("params.yaml") as f:
    model_name = yaml.safe_load(f)["bertweet"]["model_name"]
print(f"Downloading tokenizer: {model_name}")
AutoTokenizer.from_pretrained(model_name, use_fast=False)
print("Downloading model weights...")
AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
print("BERTweet download complete — cached locally.")
PYEOF
push_status "Vast.ai: BERTweet downloaded, starting step A $(date -u '+%H:%M')"

# Force HuggingFace to use only local cache — no more network calls during training
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Use only GPU 0 — multi-GPU CUDA init can hang on some hosts
export CUDA_VISIBLE_DEVICES=0

echo "--- Step A: Fine-tune BERTweet on PHEME only (~10 min) ---"
python src/compare_retraining.py --step a
push_status "Vast.ai: step-A done — PHEME-only BERTweet trained $(date -u '+%H:%M')"

echo "--- Step pseudo: Pseudo-label accumulated live tweets ---"
python src/compare_retraining.py --step pseudo
push_status "Vast.ai: pseudo-labeling done — augmented dataset ready $(date -u '+%H:%M')"

echo "--- Step B: Fine-tune BERTweet on PHEME + pseudo-labels (~25 min) ---"
python src/compare_retraining.py --step b
push_status "Vast.ai: step-B done — augmented BERTweet trained $(date -u '+%H:%M')"

echo "--- Step compare: Side-by-side comparison on held-out test set ---"
python src/compare_retraining.py --step compare
push_status "Vast.ai: comparison experiment done $(date -u '+%H:%M')"

echo "=== [5b/6] Run drift analysis (mixing experiment) ==="
python src/drift_analysis.py

echo "=== [6/6] Push final results to GitHub ==="
git fetch origin master
git rebase origin/master

# Stage all new metric files and the updated production model
git add \
    metrics/bertweet_pheme_only.json \
    metrics/bertweet_augmented.json \
    metrics/retraining_comparison.json \
    metrics/pseudo_label_stats.json \
    metrics/augmented_dataset_metadata.json \
    metrics/drift_analysis.json \
    metrics/pheme_baselines.json \
    data/processed/pheme_augmented.csv \
    2>/dev/null || true

git diff --cached --quiet || git commit -m \
    "Vast.ai: BERTweet fine-tuned on PHEME $(date -u '+%Y-%m-%d') — comparison + drift analysis"
git push origin master
trap - ERR

echo ""
echo "============================================"
echo "  DONE! Summary:"
echo "--------------------------------------------"
echo "  PHEME-only model:"
cat metrics/bertweet_pheme_only.json
echo ""
echo "  Augmented model:"
cat metrics/bertweet_augmented.json
echo ""
echo "  Comparison:"
cat metrics/retraining_comparison.json
echo "============================================"
