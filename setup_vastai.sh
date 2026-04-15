#!/bin/bash
# Run this on Vast.ai after upload_models.sh
# Usage: bash setup_vastai.sh '<MODELS_URL>'

set -e

MODELS_URL="$1"
if [ -z "$MODELS_URL" ]; then
    echo "ERROR: Provide the URL from upload_models.sh"
    echo "Usage: bash setup_vastai.sh '<URL>'"
    exit 1
fi

echo "=== [1/5] Installing system dependencies ==="
apt-get update -qq
apt-get install -y -qq git python3-pip python3-venv curl

echo "=== [2/5] Cloning repo ==="
git clone https://github.com/zegdane1998/fake-news-mlops.git
cd fake-news-mlops

echo "=== [3/5] Setting up Python environment ==="
python3 -m venv venv
source venv/bin/activate
pip install --quiet --upgrade pip

# Install requirements (Linux — no pywin32)
pip install --quiet \
    tensorflow==2.15.0 \
    fastapi uvicorn jinja2 pydantic \
    pandas numpy "scikit-learn>=1.3" \
    python-dotenv tweepy \
    dvc pytest flake8 \
    "matplotlib>=3.7" "scipy>=1.10" \
    "mlflow>=2.10" \
    "torch>=2.0" "transformers>=4.38" datasets \
    requests pyyaml

echo "=== [4/5] Downloading models and data ==="
curl -s -L "$MODELS_URL" -o models_and_data.tar.gz
tar -xzf models_and_data.tar.gz
rm models_and_data.tar.gz

echo "=== [5/5] Done! ==="
echo ""
echo "  To start the web app:"
echo "  cd fake-news-mlops && source venv/bin/activate"
echo "  uvicorn src.app:app --host 0.0.0.0 --port 8000"
echo ""
echo "  To open MLflow UI (in a separate terminal):"
echo "  cd fake-news-mlops && source venv/bin/activate"
echo "  mlflow ui --host 0.0.0.0 --port 5000"
echo ""
echo "  On Vast.ai, open the port in the instance dashboard to access from browser."
