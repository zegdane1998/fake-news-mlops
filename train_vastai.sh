#!/bin/bash
# GPU training script for Vast.ai
#
# Pipeline:
#   1. Prepare data (PHEME download + preprocessing)
#   2. Fine-tune BERTweet on PHEME only          → metrics/bertweet_pheme_only.json
#   3. Push fine-tuned model to Hugging Face Hub → abdellahzegdane/fakeshield-bertweet
#   4. Pseudo-label live tweets (fake only)
#   5. Fine-tune BERTweet on PHEME + pseudo      → metrics/bertweet_augmented.json
#   6. Side-by-side comparison                   → metrics/retraining_comparison.json
#   7. Push all results to GitHub
#
# Usage:
#   bash train_vastai.sh <GITHUB_TOKEN> [HF_TOKEN]

set -e

GITHUB_TOKEN="$1"
HF_TOKEN="$2"
REPO="zegdane1998/fake-news-mlops"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "ERROR: provide your GitHub token"
    exit 1
fi

push_status() {
    local msg="$1"
    git pull origin master --rebase --quiet 2>/dev/null || true
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
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true
pip install --quiet \
    "transformers==4.40.0" \
    datasets \
    pandas numpy "scikit-learn>=1.3" \
    pyyaml requests tqdm accelerate \
    scipy sentencepiece emoji

echo "=== [4/6] Prepare PHEME data ==="
python src/download_pheme.py
python src/preprocessing.py \
    --input  data/raw/pheme_tweets.csv \
    --output data/processed/pheme_cleaned.csv \
    --mode   tweet
push_status "Vast.ai: data ready, starting comparison experiment $(date -u '+%H:%M')"

echo "=== [5/6] Run comparison experiment ==="

echo "--- Pre-downloading BERTweet from HuggingFace (~500 MB) ---"
python - <<'PYEOF'
import yaml
with open("params.yaml") as f:
    model_name = yaml.safe_load(f)["bertweet"]["model_name"]
import emoji
print(f"emoji {emoji.__version__} OK")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
print(f"Tokenizer: {tok.__class__.__name__}  vocab_size={tok.vocab_size}")
AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
print("BERTweet download complete.")
PYEOF
push_status "Vast.ai: BERTweet downloaded, starting step A $(date -u '+%H:%M')"

# Use only GPU 0 — multi-GPU CUDA init can hang on some hosts
export CUDA_VISIBLE_DEVICES=0

echo "--- Step A: Fine-tune BERTweet on PHEME only (~10 min) ---"
python src/compare_retraining.py --step a
push_status "Vast.ai: step-A done — PHEME-only BERTweet trained $(date -u '+%H:%M')"

echo "--- Push model to Hugging Face Hub ---"
# NOTE: do this BEFORE setting TRANSFORMERS_OFFLINE
if [ -n "$HF_TOKEN" ]; then
    set +e
    HF_TOKEN="$HF_TOKEN" python - <<'PYEOF'
import os, yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer
hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token:
    print("HF_TOKEN empty — skipping")
else:
    with open("params.yaml") as f:
        model_name = yaml.safe_load(f)["bertweet"]["model_name"]
    print("Pushing to HF Hub: abdellahzegdane/fakeshield-bertweet ...")
    model = AutoModelForSequenceClassification.from_pretrained("models/bertweet_pheme_only")
    tok   = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model.push_to_hub("abdellahzegdane/fakeshield-bertweet", token=hf_token)
    tok.push_to_hub(  "abdellahzegdane/fakeshield-bertweet", token=hf_token)
    print("Model pushed to HF Hub successfully.")
PYEOF
    set -e
else
    echo "HF_TOKEN not provided — skipping HF Hub push"
fi

# Go offline AFTER the HF push so no more network calls during training
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "--- Step pseudo: Pseudo-label accumulated live tweets ---"
python src/compare_retraining.py --step pseudo
push_status "Vast.ai: pseudo-labeling done — augmented dataset ready $(date -u '+%H:%M')"

echo "--- Step B: Fine-tune BERTweet on PHEME + pseudo-labels (~25 min) ---"
python src/compare_retraining.py --step b
push_status "Vast.ai: step-B done — augmented BERTweet trained $(date -u '+%H:%M')"

echo "--- Step compare: Side-by-side comparison ---"
python src/compare_retraining.py --step compare
push_status "Vast.ai: comparison experiment done $(date -u '+%H:%M')"

echo "=== [6/6] Push final results to GitHub ==="
git fetch origin master
git rebase origin/master

git add \
    metrics/bertweet_pheme_only.json \
    metrics/bertweet_augmented.json \
    metrics/retraining_comparison.json \
    metrics/pseudo_label_stats.json \
    metrics/augmented_dataset_metadata.json \
    metrics/pheme_baselines.json \
    data/processed/pheme_augmented.csv \
    2>/dev/null || true

git commit --allow-empty -m \
    "Vast.ai: BERTweet fine-tuned on PHEME $(date -u '+%Y-%m-%d') — comparison complete"
git push origin master
trap - ERR

echo ""
echo "============================================"
echo "  DONE!"
echo "--------------------------------------------"
cat metrics/retraining_comparison.json
echo "============================================"
