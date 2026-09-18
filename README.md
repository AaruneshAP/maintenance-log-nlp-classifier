# Industrial NLP Maintenance Log Classification & SQL Analytics Pipeline

An end-to-end NLP and analytical data engineering system that ingests free-text technician maintenance logs, classifies them into failure categories using calibrated machine learning models, persists predictions in a live PostgreSQL database, exposes operational analytics via custom SQL queries and an interactive dashboard, serves low-latency REST API predictions, and automates ongoing ingestion with confidence-driven alerting.

---

## 🔗 Live Application Links & Project Artifacts

* 🚀 **Interactive Streamlit Dashboard**: [https://project3cat-8dltbsn4gedctyhueqqmf9.streamlit.app/](https://project3cat-8dltbsn4gedctyhueqqmf9.streamlit.app/)
* ⚡ **FastAPI REST API Docs (Swagger UI)**: [https://project3-cat.onrender.com/docs](https://project3-cat.onrender.com/docs)
* 🗄️ **Analytical SQL Queries Artifact**: [`queries.sql`](queries.sql) | [`results/sql_analytics_sample.md`](results/sql_analytics_sample.md)
* 📊 **Model Selection Writeup**: [`results/model_choice_rationale.md`](results/model_choice_rationale.md)
* ⚙️ **Scheduled Ingestion Workflow**: [`.github/workflows/scheduled_ingestion.yml`](.github/workflows/scheduled_ingestion.yml)
* 🔄 **Scheduled Keep-Alive Workflow**: [`.github/workflows/keep-alive.yml`](.github/workflows/keep-alive.yml)

---

## 📌 Executive Summary & Problem Statement

In modern manufacturing and processing facilities, equipment maintenance logs are generated daily by field technicians across pumps, motors, conveyers, CNC machinery, and HVAC systems. These logs consist of free-text technician observations (e.g., *"Motor DE bearing running hot at 185F with high grinding noise"*). 

Manually triaging thousands of unstructured tickets leads to:
- **Misallocated Maintenance Resources**: Misidentifying mechanical friction as electrical or thermal faults.
- **Delayed Critical Repairs**: High-severity bearing or hydraulic failures buried under low-priority routine tickets.
- **Lack of Operational Visibility**: Inability to run time-series SQL analytics to detect recurring failure modes across specific equipment IDs.

This project solves these challenges by implementing an automated NLP pipeline that classifies unstructured text logs into 8 failure categories with **94.63% test F1-score**, logs all predictions with confidence scores in a PostgreSQL database, displays live operational trends on a Streamlit dashboard, serves real-time predictions via a Dockerized FastAPI service, and continuously ingests new tickets while flagging low-confidence predictions (`confidence < 0.70`) for human review.

---

## 🏗️ System Architecture

```
                                  [ TECHNICIAN MAINTENANCE LOGS ]
                                                 │
                                                 ▼
                             [ 1. SYNTHETIC DATA GENERATION ENGINE ]
                             - Google Gemini API (gemini-flash-lite)
                             - Domain Context Constraints & Strict Dedup
                                                 │
                                                 ▼
                             [ 2. NLP PREPROCESSING & CLASSIFICATION ]
                             - TF-IDF Vectorizer (Unigrams + Bigrams)
                             - Calibrated Linear SVM (Platt Scaling)
                             - 8 Target Failure Categories
                                                 │
                                                 ▼
                             [ 3. DATABASE PERSISTENCE LAYER ]
                             - Neon PostgreSQL Database (tickets table)
                             - Schema: text, predicted_cat, confidence,
                               true_cat, equipment_id, created_at
                                                 │
                       ┌─────────────────────────┴─────────────────────────┐
                       ▼                                                   ▼
         [ 4. ANALYTICAL SQL LAYER ]                          [ 5. FASTAPI REST API ]
         - Failure Volume Over Time                           - POST /predict (DB Ingestion)
         - Low-Confidence Review Queue                        - GET  /health
         - Equipment Breakdown (Window Functions)             - Dockerized Container Deployment
                       │                                                   │
                       ▼                                                   │
         [ 6. STREAMLIT DASHBOARD ]                                        │
         - Operational Analytics (Tab 1)                                   │
         - Live Interactive Predictor (Tab 2)                              │
                       │                                                   │
                       └─────────────────────────┬─────────────────────────┘
                                                 │
                                                 ▼
                             [ 7. AUTOMATED SCHEDULED INGESTION ]
                             - GitHub Actions Scheduled Workflow (6-Hour Cron)
                             - Automatic Synthetic Batch Ingestion
                             - Low-Confidence Alerting (GitHub Issues API)
```

---

## 🛠️ Technology Stack

| Layer | Component / Technology | Purpose |
| :--- | :--- | :--- |
| **Data Generation** | Google Gemini API (`google-genai`), Python | Synthetic technician ticket generation & prompt documentation |
| **NLP & ML** | `scikit-learn`, `xgboost`, `pandas`, `numpy`, `joblib` | TF-IDF text vectorization, model training, Platt scaling calibration |
| **Database** | PostgreSQL (Neon Cloud), `SQLAlchemy`, `psycopg2-binary` | Central data warehouse for ticket storage, schema management, indexing |
| **Analytics** | SQL (PostgreSQL Window Functions & Aggregations) | Category distribution, equipment failure trends, review queue generation |
| **Dashboard** | Streamlit, Plotly Express | Interactive operational dashboard & real-time ticket classification demo |
| **REST API** | FastAPI, Uvicorn, Pydantic | Production-grade REST API endpoint for real-time model inference & DB logging |
| **Containerization**| Docker, Linux Debian slim base image | CPU-optimized container deployment for FastAPI serving layer |
| **Automation** | GitHub Actions, GitHub REST API, Python | 6-hour cron ingestion workflow, daily keep-alive scheduled commit & low-confidence triage issue alerting |

---

## 📊 Key Results & Model Selection

Three candidate classification architectures were trained and evaluated on 4,000 technician logs across a **70/15/15 stratified split** (2,800 train / 600 validation / 600 test):

### Validation Set Performance Comparison

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Calibrated)** 🏆 | **96.50%** | **96.49%** | **96.50%** | **96.50%** | **96.49%** |
| Logistic Regression | 94.50% | 94.46% | 94.54% | 94.50% | 94.46% |
| XGBoost Classifier | 90.83% | 90.79% | 90.81% | 90.83% | 90.79% |

* **Selected Model**: **Linear Support Vector Machine (`LinearSVC`)** wrapped with `CalibratedClassifierCV` (Platt Scaling) to generate calibrated probability distributions.
* **Held-Out Test Set Evaluation**: **94.67% Test Accuracy** | **94.63% Test Macro F1**.

### Per-Category Performance Breakdown (Held-Out Test Set)

| Failure Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `bearing_failure` | 0.961 | 0.987 | **0.974** | 75 |
| `corrosion` | 0.974 | 0.987 | **0.980** | 75 |
| `electrical_fault` | 0.973 | 0.960 | **0.966** | 75 |
| `hydraulic_leak` | 0.987 | 1.000 | **0.993** | 75 |
| `overheating` | 0.958 | 0.920 | **0.939** | 75 |
| `sensor_malfunction` | 0.948 | 0.973 | **0.961** | 75 |
| `software_control_fault` | 0.947 | 0.947 | **0.947** | 75 |
| `wear_and_tear` | 0.973 | 0.947 | **0.960** | 75 |

---

## 🗄️ Analytical SQL Layer

The database schema and named analytical queries in [`queries.sql`](queries.sql) demonstrate advanced SQL skills against the live PostgreSQL database:

1. **Category Volume Over Time**: Groups tickets by week and category to track failure volume shifts.
2. **Low-Confidence Human Review Queue**: Queries all tickets with `confidence < 0.70` or `true_category IS NULL` for maintenance supervisor triage.
3. **Equipment Failure Breakdown**: Aggregates top failure modes per synthetic `equipment_id`.
4. **Trending Failure Modes**: Utilizes PostgreSQL window functions (`LAG() OVER (...)`) to calculate percentage growth between consecutive 7-day windows.
5. **Database Pipeline Integrity Check**: Compares `predicted_category` against `true_category` across all ingested seed records to verify end-to-end database pipeline health (~98.68% agreement — this reflects pipeline correctness against the full dataset, including training data, and should not be confused with the 94.63% held-out test Macro F1 reported above).

---

## 🛠️ Engineering Challenges Solved

### Challenge 1: SQL Pipeline Integrity Check vs. Out-of-Sample Test Set F1 (Methodological Rigor)
* **Problem**: SQL Query 5 and Dashboard KPI 4 compute overall classifier agreement across all 4,000 database seed records, yielding ~98.68% accuracy. Technical reviewers might mistake this for an overoptimistic evaluation metric that includes training data.
* **Diagnosis & Resolution**: Explicitly separated the metrics in code and UI labels. KPI 4 was renamed to *"Pipeline Integrity Check"* with an explanatory footnote. Out-of-sample model generalization performance is documented separately on the 15% held-out test set (**94.63% Macro F1 / 94.67% Accuracy**). This distinction ensures methodological transparency during technical interviews.

### Challenge 2: PyPI Dependency Availability & Render Container Build Failure
* **Problem**: When deploying the Docker container to Render, `pip install -r requirements.txt` failed because `xgboost==3.3.0` was un-pinned locally and did not exist on PyPI at all (confirmed via the `pip` index error log listing available releases only up to `3.2.0`).
* **Diagnosis & Resolution**: Diagnosed that the saved production model artifact (`models/best_model.joblib`) was actually a `scikit-learn` Calibrated `LinearSVC` model, not XGBoost (XGBoost was only an evaluated candidate). Pinned exact stable PyPI release versions (`scikit-learn==1.6.1`, `xgboost==2.1.4`, `numpy==2.1.3`, `joblib==1.4.2`) locally, retrained the pipeline, and re-serialized artifacts natively under these versions. This completely resolved container build failures and eliminated unpickling version warnings.

### Challenge 3: GitHub Actions Repository Permissions & Automated Issue Creation
* **Problem**: The automated 6-hour ticket ingestion workflow executed successfully for database writes, but failed during GitHub Issue creation for low-confidence predictions (`confidence < 0.70`), returning `HTTP Error 403: Forbidden (Resource not accessible by integration)`.
* **Diagnosis & Resolution**: Discovered that GitHub Actions runner tokens (`${{ secrets.GITHUB_TOKEN }}`) default to read-only permissions for repository issues unless enabled at both the repository and workflow levels. Changed repository-level settings under **Settings → Actions → General → Workflow permissions** to *Read and write permissions*, and explicitly declared `permissions: issues: write` in `.github/workflows/scheduled_ingestion.yml`. The script's `try/except` wrapper ensured database writes were never interrupted by API authorization blocks.

### Challenge 4: Cross-Environment Prediction Confidence Mismatch (69% vs 54%)
* **Problem**: Testing an ambiguous ticket (*"Pump outboard bearing temperature elevated to 185F, motor overheating warning on panel"*) locally returned 69% confidence for `overheating`, but returned 54% confidence when evaluated on Render.
* **Diagnosis & Resolution**: Conducted a three-way consistency audit across local Uvicorn, Streamlit, and Render. The root cause was likely minor version discrepancies in `scikit-learn` and `numpy` across environments, affecting numerical precision somewhere in the TF-IDF vectorization or Platt scaling calibration pipeline. Pinning exact package versions (`scikit-learn==1.6.1`, `xgboost==2.1.4`, `numpy==2.1.3`, `joblib==1.4.2`) and re-saving models natively resolved the discrepancy — confirmed identical predictions across local, Streamlit, and Render for tested cases.

### Challenge 5: Streamlit Community Cloud Build Failure & Python 3.14 Default
* **Problem**: Deploying `app.py` to Streamlit Community Cloud resulted in dependency resolution hanging indefinitely with no clear error output.
* **Diagnosis & Resolution**: Identified that Streamlit Cloud had defaulted to a Python 3.14 build environment, for which C-extension libraries (`numpy==2.1.3`, `scikit-learn==1.6.1`) had no pre-compiled wheel releases. Created [`runtime.txt`](runtime.txt) in the repository root explicitly setting `python-3.11` to match the Docker container and local dev environment, resolving the dependency resolution hang.

### Challenge 6: Stale Database Connections After Neon Auto-Suspend
* **Problem**: The Streamlit dashboard's Analytics tab intermittently failed with `psycopg2.OperationalError: SSL connection has been closed unexpectedly` when querying Postgres after a period of inactivity.
* **Diagnosis & Resolution**: Traced this to Neon's serverless auto-suspend behavior — the exact tradeoff accepted when choosing Neon over Supabase for this project (scale-to-zero cost efficiency in exchange for the application layer needing to handle reconnection). SQLAlchemy was holding a connection object that Neon had silently invalidated after the compute suspended and resumed. Fixed by enabling `pool_pre_ping=True` on the SQLAlchemy engine, which transparently tests and refreshes stale connections before each query instead of raising an error.

---

## ⚠️ Known Limitations

1. **Synthetic Data Rationale**:
   * *Limitation*: No publicly accessible dataset of real-world free-text maintenance logs exists at scale due to corporate confidentiality.
   * *Mitigation*: Synthetic data generation via LLMs is a documented, defensible engineering tradeoff. Prompts were constrained with domain-specific equipment contexts (`data_generation_prompt.md`), combined with exact normalisation and near-duplicate deduplication (`difflib.SequenceMatcher > 0.85`).

2. **Domain Category Overlap (`overheating` vs `bearing_failure`)**:
   * *Limitation*: Industrial maintenance descriptions often exhibit natural physical overlap (e.g., a seized bearing causing motor overheating).
   * *Mitigation*: The system handles domain ambiguity gracefully by outputting full calibrated probability distributions and routing any ticket with max probability `< 0.70` directly to the low-confidence review queue and automated GitHub Issue alert system.

3. **Synthetic Time-Series Distribution**:
   * *Limitation*: In the synthetic dataset generation, timestamps were generated uniformly across the past 90 days.
   * *Note*: The weekly volume trend chart and window function growth metrics (`LAG() OVER (...)`) in Query 4 are illustrative of SQL query and dashboard visualization capabilities rather than physical industrial seasonal shifts.

---

## 🚀 Setup & Installation Instructions

### Prerequisites
* Python 3.11 (pinned via `runtime.txt` — newer versions may lack pre-compiled wheels for pinned ML dependencies, see Challenge 5)
* PostgreSQL database (or free-tier Neon PostgreSQL connection string)
* Docker (optional, for containerized API serving)

### 1. Repository Setup & Environment
```bash
# Clone the repository
git clone https://github.com/AaruneshAP/Project3_cat.git
cd Project3_cat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```env
DATABASE_URL=postgresql://username:password@ep-sample-pooler.c-3.aws.neon.tech/neondb?sslmode=require
GEMINI_API_KEY=your_google_gemini_api_key
```

### 3. Run Pipeline Components Locally
```bash
# Train ML models and evaluate candidates
python train_classifier.py

# Seed PostgreSQL database with predictions
python load_predictions.py

# Launch Streamlit Analytics Dashboard
streamlit run app.py

# Launch FastAPI REST API server
uvicorn main:app --reload --port 8000
```

### 4. Run Docker Container Locally
```bash
# Build Docker image
docker build -t nlp-maintenance-api .

# Run container exposing port 8000
docker run -p 8000:8000 --env-file .env nlp-maintenance-api
```

### 5. Streamlit Community Cloud Deployment & Keep-Alive Automation

* **Live Dashboard URL**: [https://project3cat-8dltbsn4gedctyhueqqmf9.streamlit.app/](https://project3cat-8dltbsn4gedctyhueqqmf9.streamlit.app/)

#### 💡 Streamlit Cloud Sleep Detection & Keep-Alive Strategy
Streamlit Community Cloud automatically hibernates inactive applications to conserve free-tier computing capacity. Plain HTTP pings (such as periodic `curl` health check requests) do **not** reset Streamlit Cloud's sleep timer — its hibernation detector requires an **actual active WebSocket session or a repository git commit**, meaning stateless HTTP requests return `200 OK` while the app still goes to sleep.

For this reason, the automated keep-alive mechanism ([`.github/workflows/keep-alive.yml`](.github/workflows/keep-alive.yml)) is implemented as a **daily scheduled commit** (`06:00 UTC`) rather than an HTTP ping script. The workflow updates the timestamp file [`.github/keepalive/last-ping.txt`](.github/keepalive/last-ping.txt) and pushes a commit via `EndBug/add-and-commit` using GitHub Actions' built-in `GITHUB_TOKEN`. Streamlit Cloud detects this repository commit and resets its inactivity timer, keeping the dashboard continuously awake and instantly accessible for recruiters and interviewers.
