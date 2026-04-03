# Thesis Diagrams — Fake News Detection MLOps

---

## 1. Overall System Architecture

```mermaid
graph TB
    subgraph DATA["Data Layer"]
        DS1[FakeNewsNet / GossipCop]
        DS2[WELFake / PHEME\nfull text — planned]
        SCR[X/Twitter Scraper\nrealtime_scraper.py]
    end

    subgraph PIPELINE["DVC Pipeline"]
        ING[ingest\ningestion.py]
        PRE[preprocess\npreprocessing.py]
        TRN[train\ntrain.py]
        EVL[evaluate\nevaluate.py]
        BAS[baselines\nbaselines.py]
    end

    subgraph MODELS["Model Registry"]
        CNN[TextCNN\ncnn_v1.h5]
        TOK[Tokenizer\ntokenizer.pkl]
        LR[TF-IDF + LogReg\ntfidf_logreg.pkl]
        SVM[TF-IDF + SVM\ntfidf_svm.pkl]
        LSTM[LSTM\nlstm_v1.h5]
        BERT[BERTweet / RoBERTa\nplanned — TRUBA]
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
    end

    DS1 --> ING
    DS2 -.->|planned| ING
    SCR --> MON
    ING --> PRE --> TRN --> CNN & TOK
    PRE --> BAS --> LR & SVM & LSTM
    TRN --> EVL
    CNN & TOK --> EVL
    CNN & TOK & LR & SVM & LSTM --> MLF
    CNN & TOK --> API --> UI
    MON --> REP
    MON -->|drift sys.exit 1| GHA2
    GHA2 -->|retrain| ING
    GHA1 -->|pytest on push| TRN
```

---

## 2. DVC Pipeline DAG

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
    PLOTS[(metrics/\nroc_curve.png\nconfidence_histogram.png\nerror_analysis.json)]

    ING["ingest\n─────────\npython src/ingestion.py"]
    PRE["preprocess\n─────────\npython src/preprocessing.py"]
    TRN["train\n─────────\npython src/train.py\nparams.yaml → train.*"]
    EVL["evaluate\n─────────\npython src/evaluate.py"]
    BAS["baselines\n─────────\npython src/baselines.py\nparams.yaml → baselines.*"]

    ING --> RAW --> PRE --> PROC
    PROC --> TRN --> CNN_H5 & TOK_PK
    CNN_H5 & TOK_PK & PROC --> EVL --> SCORES & PLOTS
    PROC & TOK_PK & SCORES --> BAS --> BAS_JSON & LR_PK & SVM_PK & LSTM_H5
```

---

## 3. TextCNN Architecture

```mermaid
graph TB
    IN["Input\nsequence length = 100"]
    EMB["Embedding\nvocab=30000 × dim=128"]

    subgraph CONV["Parallel Convolutions"]
        C3["Conv1D kernel=3\n128 filters → GlobalMaxPool"]
        C4["Conv1D kernel=4\n128 filters → GlobalMaxPool"]
        C5["Conv1D kernel=5\n128 filters → GlobalMaxPool"]
    end

    CONCAT["Concatenate\n384-dim"]
    DROP1["Dropout 0.5"]
    DENSE["Dense 128  ReLU"]
    DROP2["Dropout 0.3"]
    OUT["Dense 1  Sigmoid\n→ P(real)"]

    IN --> EMB
    EMB --> C3 & C4 & C5
    C3 & C4 & C5 --> CONCAT --> DROP1 --> DENSE --> DROP2 --> OUT
```

---

## 4. Model Comparison Hierarchy (Thesis Contribution)

```mermaid
graph LR
    subgraph CLASSICAL["Classical ML"]
        LR2["TF-IDF + LogReg"]
        SVM2["TF-IDF + SVM\nCalibratedClassifierCV"]
    end

    subgraph DEEP["Deep Learning"]
        LSTM2["LSTM\nsingle-layer"]
        CNN2["TextCNN\nKim 2014\nmulti-filter"]
    end

    subgraph TRANSFORMER["Transformers — TRUBA GPU"]
        BT["BERTweet\ntwitter-specific"]
        RB["RoBERTa-base"]
        DB["DeBERTa-v3"]
    end

    CLASSICAL -->|baseline| DEEP
    DEEP -->|scale up| TRANSFORMER

    note1["Headlines only\nFakeNewsNet"] -.-> CLASSICAL & DEEP
    note2["Full tweets\nWELFake / PHEME"] -.-> TRANSFORMER
```

---

## 5. Continuous Training (CT) Pipeline — GitHub Actions

```mermaid
sequenceDiagram
    participant SCH as Scheduler<br/>(daily 06:00 UTC)
    participant GHA as GitHub Actions
    participant SCR as realtime_scraper.py
    participant MON as monitor.py
    participant DVC as DVC Pipeline
    participant GIT as Git / DVC Remote

    SCH->>GHA: trigger scrape_and_monitor.yml
    GHA->>SCR: python src/realtime_scraper.py
    SCR-->>GHA: data/new_scraped/*.csv

    GHA->>MON: python src/monitor.py
    MON->>MON: relevancy check (keywords)
    MON->>MON: KS-test + PSI vs reference dist.

    alt No drift (exit 0)
        MON-->>GHA: STABLE
        GHA->>GIT: commit drift_report.json
    else Drift detected (exit 1)
        MON-->>GHA: DRIFT TRIGGERED
        GHA->>DVC: ingestion → preprocess → train → evaluate
        DVC-->>GHA: new cnn_v1.h5 + metrics
        GHA->>GIT: commit models/ metrics/ data/new_scraped/
    end
```

---

## 6. Monitoring & Drift Detection Logic

```mermaid
flowchart TD
    START([New scraped data arrives])
    REL{Relevancy rate\n≥ 30%?}
    CONF{Avg confidence\n≥ 0.30?}
    KS{KS-test\np ≥ 0.05?}
    PSI{PSI < 0.25?}

    STABLE([STABLE\nsys.exit 0])
    RETRAIN([RETRAIN TRIGGERED\nsys.exit 1])

    START --> REL
    REL -->|No| RETRAIN
    REL -->|Yes| CONF
    CONF -->|No| RETRAIN
    CONF -->|Yes| KS
    KS -->|No — drift| RETRAIN
    KS -->|Yes — stable| PSI
    PSI -->|No — PSI ≥ 0.25| RETRAIN
    PSI -->|Yes — PSI < 0.10 STABLE\n0.10–0.25 MODERATE| STABLE

    RETRAIN --> RT["Re-run DVC pipeline\ningestion → preprocess\n→ train → evaluate"]
```

---

## 7. Serving Architecture — FastAPI

```mermaid
graph LR
    USER([User / Browser])
    API["FastAPI app.py\nlocalhost:8000"]
    CNN3["cnn_v1.h5\n+ tokenizer.pkl\n(loaded at startup)"]
    SCRAPE["data/new_scraped/*.csv\n(latest file)"]
    TMPL["Jinja2 Template\nindex.html"]

    USER -->|GET /| API
    USER -->|POST /analyze\nheadline text| API
    API --> CNN3
    API --> SCRAPE
    API --> TMPL --> USER

    subgraph INFERENCE["Inference Steps"]
        T1["tokenize → pad\n(MAX_LEN=100)"]
        T2["model.predict"]
        T3["verdict: Real / Fake\n+ confidence %"]
        T1 --> T2 --> T3
    end

    CNN3 --> INFERENCE
```

---

## 8. Planned TRUBA Scale-Up Flow

```mermaid
flowchart TD
    subgraph LOCAL["Local Machine"]
        CODE["Updated DVC pipeline\n+ HuggingFace Trainer script\ntrain_transformer.py"]
        PUSH["git push + dvc push"]
    end

    subgraph TRUBA["TRUBA HPC Cluster"]
        SLURM["SLURM job submission\n#SBATCH --gres=gpu:1"]
        ENV["Python venv\ntransformers + datasets"]
        TRF["Fine-tune\nBERTweet / RoBERTa / DeBERTa-v3\non WELFake / PHEME full text"]
        CKPT["Checkpoint saved\nmodels/berttweet_v1/"]
    end

    subgraph RESULTS["Results → Thesis"]
        MLF2["MLflow\nexperiment comparison"]
        TABLE["Comparison Table\nClassical → LSTM → CNN → Transformer\nAcc / F1 / AUC-ROC"]
    end

    CODE --> PUSH --> TRUBA
    SLURM --> ENV --> TRF --> CKPT
    CKPT --> MLF2 --> TABLE
```
