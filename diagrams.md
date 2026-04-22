# Thesis Diagrams — Fake News Detection MLOps

---

## 1. Overall System Architecture

```mermaid
graph TB
    subgraph DATA["Data Layer"]
        DS1[FakeNewsNet / GossipCop\nheadlines]
        DS2[PHEME\ntweet threads]
        SCR[X/Twitter Scraper\nrealtime_scraper.py]
    end

    subgraph PIPELINE["DVC Pipeline — FakeNewsNet"]
        ING[ingest\ningestion.py]
        PRE[preprocess\npreprocessing.py]
        TRN[train\ntrain.py]
        EVL[evaluate\nevaluate.py]
        BAS[baselines\nbaselines.py]
    end

    subgraph VAST["Vast.ai Pipeline — PHEME"]
        V_DOWN[download_pheme.py]
        V_PRE[preprocessing.py\ntweet mode]
        V_BERT[train_bertweet.py\n2× RTX 3090]
    end

    subgraph MODELS["Model Registry"]
        CNN[TextCNN\ncnn_v1.h5\nAcc 0.81]
        LR[TF-IDF + LogReg\nAcc 0.84]
        SVM[TF-IDF + SVM\nAcc 0.85]
        LSTM[LSTM\nAcc 0.84]
        BERT[BERTweet\nbertweet_finetuned/\nAcc 0.96 ✓]
    end

    subgraph TRACKING["Experiment Tracking"]
        MLF[MLflow]
    end

    subgraph MONITOR["Monitoring"]
        MON[monitor.py\nKS-test + PSI]
        REP[drift_report.json]
    end

    subgraph SERVE["Serving"]
        API[FastAPI\napp.py]
        UI[Web Dashboard]
    end

    subgraph CI["CI/CD — GitHub Actions"]
        GHA1[Python CI\nmain.yml]
        GHA2[Scrape & Monitor\nscrape_and_monitor.yml\ndaily 06:00 UTC]
        GHA3[Retrain on Vast.ai\nretrain_vastai.yml\nmanual trigger]
    end

    DS1 --> ING
    DS2 --> V_DOWN --> V_PRE --> V_BERT --> BERT
    SCR --> MON
    ING --> PRE --> TRN --> CNN
    PRE --> BAS --> LR & SVM & LSTM
    TRN --> EVL
    CNN & LR & SVM & LSTM & BERT --> MLF
    CNN --> API --> UI
    MON --> REP
    MON -->|drift| GHA2 -->|retrain| ING
    GHA1 -->|pytest on push| TRN
    GHA3 --> V_DOWN
```

---

## 2. DVC Pipeline DAG — FakeNewsNet

```mermaid
flowchart LR
    RAW[(data/raw/\ngossipcop_combined.csv)]
    PROC[(data/processed/\ngossipcop_cleaned.csv)]
    CNN_H5[(models/\ncnn_v1.h5)]
    TOK_PK[(models/\ntokenizer.pkl)]
    SCORES[(metrics/\nscores.json)]
    BAS_JSON[(metrics/\nbaselines.json)]
    LR_PK[(models/\ntfidf_logreg.pkl)]
    SVM_PK[(models/\ntfidf_svm.pkl)]
    LSTM_H5[(models/\nlstm_v1.h5)]
    PLOTS[(metrics/\nroc_curve.png\nconfidence_histogram.png)]

    ING["ingest\n─────────\npython src/ingestion.py"]
    PRE["preprocess\n─────────\npython src/preprocessing.py"]
    TRN["train\n─────────\npython src/train.py"]
    EVL["evaluate\n─────────\npython src/evaluate.py"]
    BAS["baselines\n─────────\npython src/baselines.py"]

    ING --> RAW --> PRE --> PROC
    PROC --> TRN --> CNN_H5 & TOK_PK
    CNN_H5 & TOK_PK & PROC --> EVL --> SCORES & PLOTS
    PROC & TOK_PK & SCORES --> BAS --> BAS_JSON & LR_PK & SVM_PK & LSTM_H5
```

---

## 3. Vast.ai BERTweet Training Pipeline

```mermaid
flowchart TD
    subgraph GHA["GitHub Actions — retrain_vastai.yml"]
        TRIGGER([Manual trigger\nworkflow_dispatch])
        LAUNCH[Find cheapest offer\n2× RTX 3090\nLaunch instance]
        VERIFY[Verify instance running\nREST API polling]
        POLL[Poll master commits\nevery 2 min — up to 180 min]
        DESTROY[Destroy instance\nalways]
    end

    subgraph VAST["Vast.ai Instance\npytorch/pytorch:2.1.0-cuda11.8"]
        SETUP[apt-get + git-lfs\ngit clone repo]
        DEPS[pip install\ntransformers datasets\nmlflow accelerate]
        DOWN[download_pheme.py\nfigshare API]
        PREP[preprocessing.py\ntweet mode\nnormalise + clean]
        STATUS1[push_status\ntraining started]
        STATUS2[push_status\ndata ready]
        TRAIN[train_bertweet.py\nvinai/bertweet-base\n5 epochs · batch 64\nDataParallel 2× GPU]
        PUSH[git push\nmetrics/bertweet_scores.json\nmetrics/baselines.json]
    end

    TRIGGER --> LAUNCH --> VERIFY --> POLL
    LAUNCH -.->|onstart-cmd| SETUP
    SETUP --> STATUS1 --> DEPS --> DOWN --> PREP --> STATUS2 --> TRAIN --> PUSH
    PUSH -->|commit: BERTweet fine-tuned on PHEME| POLL
    POLL -->|detected success| DESTROY
    POLL -->|FAILED commit detected| DESTROY
```

---

## 4. TextCNN Architecture

```mermaid
graph TB
    IN["Input\nsequence length = 100"]
    EMB["Embedding\nvocab=30 000 × dim=128"]

    subgraph CONV["Parallel Convolutions"]
        C3["Conv1D kernel=3\n128 filters → GlobalMaxPool"]
        C4["Conv1D kernel=4\n128 filters → GlobalMaxPool"]
        C5["Conv1D kernel=5\n128 filters → GlobalMaxPool"]
    end

    CONCAT["Concatenate  384-dim"]
    DROP1["Dropout 0.5"]
    DENSE["Dense 128  ReLU"]
    DROP2["Dropout 0.3"]
    OUT["Dense 1  Sigmoid → P(real)"]

    IN --> EMB
    EMB --> C3 & C4 & C5
    C3 & C4 & C5 --> CONCAT --> DROP1 --> DENSE --> DROP2 --> OUT
```

---

## 5. BERTweet Fine-Tuning Architecture

```mermaid
graph TB
    IN2["Input tweet\nnormalised: HTTPURL @USER tokens"]
    TOK2["BERTweet Tokenizer\nvinai/bertweet-base\nmax_len=128  use_fast=False"]

    subgraph BERT_BODY["BERTweet Encoder  110M params"]
        EMB2["Token + Position Embeddings\ndim=768"]
        TR["12× Transformer Layers\nMulti-Head Self-Attention"]
        CLS["[CLS] representation\n768-dim"]
    end

    subgraph HEAD["Classification Head"]
        DROP3["Dropout 0.1"]
        LIN["Linear 768 → 2"]
        SOFT["Softmax → P(fake) / P(real)"]
    end

    TRAIN2["Training\nAdamW lr=2e-5  wd=0.01\n5 epochs  batch=64  warmup=10%\nDataParallel — 2× RTX 3090"]

    IN2 --> TOK2 --> EMB2 --> TR --> CLS --> DROP3 --> LIN --> SOFT
    TRAIN2 -.->|fine-tune| BERT_BODY
```

---

## 6. Model Comparison — All Results

```mermaid
graph LR
    subgraph CLASSICAL["Classical ML — FakeNewsNet headlines"]
        LR2["TF-IDF + LogReg\nAcc 0.837  F1 0.790\nAUC 0.882"]
        SVM2["TF-IDF + SVM\nAcc 0.853  F1 0.779\nAUC 0.876"]
    end

    subgraph DEEP["Deep Learning — FakeNewsNet headlines"]
        LSTM2["LSTM\nAcc 0.841  F1 0.776\nAUC 0.869"]
        CNN2["TextCNN\nAcc 0.813  F1 0.770\nAUC 0.873"]
    end

    subgraph TRANSFORMER["Transformers — PHEME tweets  ✓ Done"]
        BT["BERTweet\nAcc 0.962  F1 0.897\nAUC 0.960\nVast.ai  2× RTX 3090"]
        RB["RoBERTa-base\nplanned"]
    end

    CLASSICAL -->|scale up| DEEP
    DEEP -->|scale up| TRANSFORMER

    note1["Headlines only"] -.-> CLASSICAL & DEEP
    note2["Full tweet threads\nPHEME dataset"] -.-> TRANSFORMER
```

---

## 7. Continuous Training Pipeline — GitHub Actions

```mermaid
sequenceDiagram
    participant SCH as Scheduler<br/>(daily 06:00 UTC)
    participant GHA as GitHub Actions
    participant SCR as realtime_scraper.py
    participant MON as monitor.py
    participant DVC as DVC Pipeline
    participant VAST as Vast.ai
    participant GIT as Git / master

    SCH->>GHA: trigger scrape_and_monitor.yml
    GHA->>SCR: python src/realtime_scraper.py
    SCR-->>GHA: data/new_scraped/*.csv

    GHA->>MON: python src/monitor.py
    MON->>MON: relevancy + KS-test + PSI

    alt No drift
        MON-->>GHA: STABLE
        GHA->>GIT: commit drift_report.json
    else Drift detected
        MON-->>GHA: DRIFT
        GHA->>DVC: ingest → preprocess → train → evaluate
        DVC-->>GHA: new cnn_v1.h5 + metrics
        GHA->>GIT: commit models/ metrics/
    end

    Note over GHA,VAST: Manual trigger — retrain_vastai.yml
    GHA->>VAST: launch 2× RTX 3090 instance
    VAST->>VAST: download PHEME → preprocess<br/>fine-tune BERTweet  5 epochs
    VAST->>GIT: commit bertweet_scores.json
    GHA->>VAST: destroy instance
```

---

## 8. Monitoring & Drift Detection Logic

```mermaid
flowchart TD
    START([New scraped data arrives])
    REL{Relevancy rate\n≥ 30%?}
    CONF{Avg confidence\n≥ 0.30?}
    KS{KS-test\np ≥ 0.05?}
    PSI{PSI < 0.25?}

    STABLE([STABLE — sys.exit 0])
    RETRAIN([RETRAIN TRIGGERED — sys.exit 1])

    START --> REL
    REL -->|No| RETRAIN
    REL -->|Yes| CONF
    CONF -->|No| RETRAIN
    CONF -->|Yes| KS
    KS -->|No — drift| RETRAIN
    KS -->|Yes — stable| PSI
    PSI -->|No — PSI ≥ 0.25| RETRAIN
    PSI -->|Yes| STABLE

    RETRAIN --> RT["Re-run DVC pipeline\ningest → preprocess → train → evaluate"]
```

---

## 9. Serving Architecture — FastAPI

```mermaid
graph LR
    USER([User / Browser])
    API["FastAPI app.py\nlocalhost:8000"]
    CNN3["cnn_v1.h5\n+ tokenizer.pkl\nloaded at startup"]
    SCRAPE["data/new_scraped/*.csv\nlatest file"]
    TMPL["Jinja2 Template\nindex.html"]

    USER -->|GET /| API
    USER -->|POST /analyze  headline| API
    API --> CNN3
    API --> SCRAPE
    API --> TMPL --> USER

    subgraph INFERENCE["Inference Steps"]
        T1["tokenize → pad\nMAX_LEN=100"]
        T2["model.predict"]
        T3["verdict: Real / Fake\n+ confidence %"]
        T1 --> T2 --> T3
    end

    CNN3 --> INFERENCE
```
