#!/bin/bash
#SBATCH --job-name=fake-news-bertweet
#SBATCH --account=azegdane
#SBATCH --partition=barbun-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/bertweet_%j.out
#SBATCH --error=logs/bertweet_%j.err

# ── Environment ────────────────────────────────────────────────────────────────
module load comp/python/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fake-news

PROJECT_DIR=/arf/scratch/azegdane/fake-news-mlops
cd "$PROJECT_DIR"
mkdir -p logs

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time : $(date)"

# ── Run pipeline ───────────────────────────────────────────────────────────────
python src/download_pheme.py
python src/preprocessing.py --input data/raw/pheme_tweets.csv \
                             --output data/processed/pheme_cleaned.csv \
                             --mode tweet
python src/train_bertweet.py

echo "Done: $(date)"

# ── Push metrics back to GitHub ────────────────────────────────────────────────
git config user.name "TRUBA-runner"
git config user.email "truba@fake-news-mlops"
git pull origin master --rebase
git add metrics/bertweet_scores.json metrics/baselines.json
git commit -m "TRUBA: BERTweet results $(date -u '+%Y-%m-%d')" || echo "Nothing to commit"
git push origin master
