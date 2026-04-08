#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# truba_cron.sh — Runs hourly on TRUBA login node via user crontab.
#
# Install once on TRUBA:
#   chmod +x /arf/scratch/azegdane/fake-news-mlops/truba_cron.sh
#   crontab -e
#   # add this line:
#   0 * * * * /arf/scratch/azegdane/fake-news-mlops/truba_cron.sh >> /arf/scratch/azegdane/fake-news-mlops/logs/cron.log 2>&1
#
# What it does (each hour):
#   1. If a completed SLURM job left a .job_done flag → push metrics to GitHub
#   2. If drift_report.json has retrain_needed=true and no job running → sbatch
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT=/arf/scratch/azegdane/fake-news-mlops
LOCK=$PROJECT/.retrain_submitted

module load comp/python/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fake-news

cd "$PROJECT"
mkdir -p logs
echo "=== $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="

# ── Step 1: push metrics if training just finished ─────────────────────────────
if [ -f "$PROJECT/.job_done" ]; then
    echo "Job completed — pushing metrics to GitHub..."
    git pull origin master --rebase
    git add metrics/bertweet_scores.json metrics/baselines.json || true
    git commit -m "TRUBA: BERTweet results $(date -u '+%Y-%m-%d')" || echo "Nothing to commit"
    git push origin master
    rm -f "$PROJECT/.job_done" "$LOCK"
    echo "Metrics pushed. Lock cleared."
fi

# ── Step 2: check if retrain is needed ────────────────────────────────────────
if [ -f "$LOCK" ]; then
    echo "Retrain already submitted — skipping."
    exit 0
fi

RETRAIN=$(curl -sf \
    "https://raw.githubusercontent.com/zegdane1998/fake-news-mlops/master/metrics/drift_report.json" \
    | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('retrain_needed', False))")

echo "retrain_needed: $RETRAIN"

if [ "$RETRAIN" = "True" ]; then
    echo "Drift detected — pulling latest code and submitting SLURM job..."
    git pull origin master --rebase
    mkdir -p logs
    JOB_ID=$(sbatch truba_job.sh | awk '{print $4}')
    echo "Submitted SLURM job $JOB_ID"
    touch "$LOCK"
else
    echo "No drift — nothing to do."
fi
