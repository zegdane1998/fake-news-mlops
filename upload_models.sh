#!/bin/bash
# Run this LOCALLY before going to Vast.ai
# Packs models + data and uploads to transfer.sh (free, valid 14 days)
# Usage: bash upload_models.sh

set -e

echo "=== Packing models and data ==="
# mlruns is 477MB — too large. Metrics are already in metrics/*.json
tar -czf models_and_data.tar.gz \
    models/cnn_v1.h5 \
    models/tokenizer.pkl \
    models/tfidf_logreg.pkl \
    models/tfidf_svm.pkl \
    models/lstm_v1.h5 \
    data/processed/gossipcop_cleaned.csv \
    metrics/scores.json \
    metrics/baselines.json \
    metrics/roc_curve.png \
    metrics/confidence_histogram.png \
    metrics/error_analysis.json

echo "=== Uploading to transfer.sh (~100MB, may take a minute) ==="
URL=$(curl -s --upload-file models_and_data.tar.gz https://transfer.sh/models_and_data.tar.gz)

echo ""
echo "============================================"
echo "  UPLOAD DONE! Copy this URL:"
echo "  $URL"
echo "============================================"
echo ""
echo "  On Vast.ai run:"
echo "  bash setup_vastai.sh '$URL'"
echo ""

rm models_and_data.tar.gz
