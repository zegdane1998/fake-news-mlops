# FakeShield — End-to-End MLOps Pipeline for Real-Time Fake News Detection

An end-to-end MLOps pipeline for fake news detection, built around a BERTweet-based
classifier with automated drift monitoring and cloud-triggered retraining. Designed
to demonstrate a production-style ML workflow, not just a trained model: data
versioning, experiment tracking, CI/CD, drift detection, and automated retraining
are all wired together.

## Why this exists

Most fake-news classifiers are trained once and evaluated on a static test set.
In the real world, language on social media shifts constantly — new events, new
slang, new patterns of misinformation — and a model trained on last year's data
quietly degrades. This project builds the operational layer around a classifier
so it keeps working as the data distribution changes, rather than assuming a
one-time training run is enough.

## What it does

- **Trains and benchmarks multiple models** for fake news classification —
  TF-IDF + Logistic Regression, TF-IDF + SVM, LSTM, TextCNN, and a fine-tuned
  BERTweet transformer — on the PHEME dataset, to compare classical and deep
  learning approaches on the same evaluation pipeline.
- **Tracks every experiment** (parameters, metrics, model artifacts) with MLflow,
  so results are reproducible and comparable across runs.
- **Versions data and models** with DVC, keeping large files out of Git while
  keeping every pipeline stage reproducible via `dvc.yaml` / `dvc.lock`.
- **Monitors for data drift** in production using statistical tests (PSI,
  Kolmogorov–Smirnov) to detect when incoming data starts to diverge from the
  training distribution.
- **Retrains automatically** via GitHub Actions when drift crosses a threshold,
  using semi-supervised continual learning with pseudo-labeling to incorporate
  new data without requiring full manual re-labeling.
- **Trains on demand in the cloud** via `train_vastai.sh`, which provisions GPU
  training on Vast.ai (RTX 3090) rather than requiring local GPU hardware.
- **Serves a lightweight monitoring dashboard** (HTML/CSS templates) to visualize
  model performance and drift metrics over time.

## Architecture

```
Data Collection ──▶ Preprocessing ──▶ Training ──▶ Evaluation
                                          │
                                          ▼
                                    MLflow Tracking
                                          │
                                          ▼
                              ┌── Drift Monitoring (PSI/KS) ──┐
                              │                                │
                        No drift detected              Drift detected
                              │                                │
                          Serve model            Cloud-triggered retraining
                                                   (GitHub Actions + Vast.ai)
                                                              │
                                                              ▼
                                                  Semi-supervised pseudo-labeling
                                                              │
                                                              ▼
                                                       Updated model ──▶ Serve
```

## Tech stack

| Layer | Tools |
|---|---|
| Modeling | HuggingFace Transformers (BERTweet), PyTorch, Scikit-learn |
| Experiment tracking | MLflow |
| Data/model versioning | DVC |
| CI/CD & automation | GitHub Actions |
| Cloud training | Vast.ai (GPU on demand) |
| Monitoring | Custom drift detection (PSI, KS statistic) |
| Dashboard | Flask, HTML/CSS |
| Dataset | PHEME |

## Project structure

```
├── src/                # Core pipeline code (preprocessing, training, monitoring)
├── models/              # Trained model artifacts
├── metrics/              # Evaluation outputs and drift metrics
├── data/                 # DVC-tracked datasets
├── templates/, static/   # Monitoring dashboard front-end
├── tests/                 # Test suite
├── paper/                 # Accompanying research write-up
├── .github/workflows/     # CI/CD and cloud-triggered retraining pipelines
├── dvc.yaml, dvc.lock      # Pipeline definition and versioning
├── params.yaml              # Central configuration for pipeline stages
└── train_vastai.sh            # Cloud GPU training script
```

## Getting started

```bash
# Clone the repo
git clone https://github.com/zegdane1998/fake-news-mlops.git
cd fake-news-mlops

# Install dependencies
pip install -r requirements.txt

# Pull DVC-tracked data and models
dvc pull

# Reproduce the pipeline
dvc repro

# Or train on a cloud GPU instance
bash train_vastai.sh
```

## Results

Benchmarked across five model architectures on the PHEME dataset, with BERTweet
outperforming classical baselines (TF-IDF + LogReg/SVM) and other deep learning
approaches (LSTM, TextCNN) on held-out evaluation metrics. See `metrics/` and the
accompanying paper for full results.

## Status

Actively developed as part of an MSc thesis on MLOps pipelines for real-time
misinformation detection. Core training, monitoring, and CI/CD components are
functional; retraining automation and dashboard are in progress.

## Author

Abdellah Zegdane — [Medium](https://medium.com/@zegdane1998) ·
[GitHub](https://github.com/zegdane1998)
