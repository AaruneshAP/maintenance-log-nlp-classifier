"""
simulate_ingestion.py
======================
Stage 9: Automated Ticket Ingestion & Low-Confidence Alerting Script

Workflow:
1. Generate a small batch of realistic synthetic maintenance logs (~12-15 tickets) via Google Gemini API
2. Classify each ticket using saved model artifacts in models/
3. Batch-insert predictions into PostgreSQL `tickets` table with `true_category = NULL`
4. Filter newly inserted tickets with `confidence < 0.70`
5. Create a GitHub Issue alert summarizing low-confidence tickets if running in CI/CD (or log locally)
"""

import os
import re
import sys
import random
import urllib.request
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from google import genai

# Reconfigure stdout for UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Setup & Config
# -----------------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY")  # e.g., "AaruneshAP/Project3_cat"

MODELS_DIR = Path("models")
EQUIPMENT_POOL = [f"EQ-{i}" for i in range(101, 121)]
CATEGORIES = [
    "bearing_failure",
    "hydraulic_leak",
    "electrical_fault",
    "overheating",
    "corrosion",
    "sensor_malfunction",
    "software_control_fault",
    "wear_and_tear"
]

# -----------------------------------------------------------------------------
# Helper: Text Preprocessing
# -----------------------------------------------------------------------------
def clean_text(text_val: str) -> str:
    if not isinstance(text_val, str):
        return ""
    t = text_val.lower()
    t = re.sub(r"[^\w\s-]", " ", t)
    t = re.sub(r"\s-\s", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# -----------------------------------------------------------------------------
# 1. Synthetic Ticket Generation (Gemini API with Fallback)
# -----------------------------------------------------------------------------
FALLBACK_TICKETS = [
    "Primary exhaust fan motor DE bearing housing running hot at 195F with high vibration and loud grinding noise.",
    "Hydraulic press main cylinder seal blown, fluid leaking rapidly onto shop floor under high pressure.",
    "Main circuit breaker tripped on overcurrent fault, control panel displays ground fault alarm.",
    "Pump outboard bearing temperature elevated to 185F, motor overheating warning on panel.",
    "Rotary kiln shell surface temperature exceeded 450C near burner head due to refractory brick erosion.",
    "Conveyor photoeye optical lens clouded with soot, throwing intermittent false blockage trip alarms.",
    "PLC rack 2 communication module offline, Ethernet/IP bus fault causing line stoppage.",
    "Compressor air intake filter housing severely corroded and rusted through near flange joint.",
    "Pneumatic cylinder rod seals worn out, air pressure leaking continuously during stroke cycle.",
    "Cooling water pump mechanical seal leaking water into oil reservoir, oil emulsified.",
    "Elevator hoist motor thermal overload trip, motor housing hot to touch after extended run.",
    "Vibration sensor accelerometer cable severed by falling scrap material on stamp line."
]

def generate_synthetic_tickets(count: int = 12) -> List[str]:
    """Generate realistic technician logs using Gemini API or fallback."""
    if not GEMINI_API_KEY:
        print("[WARNING] GEMINI_API_KEY not found. Using local fallback synthetic batch.")
        return FALLBACK_TICKETS[:count]
    
    print(f"Connecting to Gemini API to generate {count} new maintenance tickets...")
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""Generate exactly {count} short, realistic industrial technician maintenance log descriptions.
Each description must be 1-2 sentences describing an equipment problem (e.g. bearing noise, hydraulic leak, electrical fault, overheating, sensor trip, PLC error, corrosion, seal wear).
Include 2 tickets that are ambiguous between overheating and bearing friction to test model triage flagging.

Output format: Return ONLY the raw ticket texts, one per line. Do NOT include numbers, bullet points, markdown formatting, or headers."""
        
        response = client.models.generate_content(
            model="models/gemini-flash-lite-latest",
            contents=prompt
        )
        
        lines = [ln.strip() for ln in response.text.strip().splitlines() if ln.strip()]
        # Strip leading numbers if any (e.g. "1. ")
        tickets = [re.sub(r"^\d+[\.\)]\s*", "", ln) for ln in lines if len(ln) > 10]
        
        if len(tickets) >= count:
            print(f"Successfully generated {len(tickets)} synthetic tickets from Gemini API.")
            return tickets[:count]
        else:
            print(f"Gemini returned {len(tickets)} tickets. Supplementing with fallback batch...")
            return (tickets + FALLBACK_TICKETS)[:count]
            
    except Exception as e:
        print(f"[WARNING] Gemini API call failed: {e}. Using fallback synthetic batch.")
        return FALLBACK_TICKETS[:count]

# -----------------------------------------------------------------------------
# 2. Model Inference & Classification
# -----------------------------------------------------------------------------
def classify_tickets(texts: List[str]) -> pd.DataFrame:
    """Classify input text logs using saved model artifacts."""
    print("Loading ML model artifacts from models/...")
    model_path = MODELS_DIR / "best_model.joblib"
    vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    enc_path = MODELS_DIR / "label_encoder.joblib"

    if not (model_path.exists() and vec_path.exists() and enc_path.exists()):
        raise FileNotFoundError("ML model artifacts missing in `models/` directory.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    encoder = joblib.load(enc_path)

    cleaned_texts = [clean_text(t) for t in texts]
    X_vec = vectorizer.transform(cleaned_texts)

    probs = model.predict_proba(X_vec)
    pred_indices = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    predicted_cats = encoder.inverse_transform(pred_indices)

    df = pd.DataFrame({
        "text": texts,
        "predicted_category": predicted_cats,
        "confidence": np.round(confidences, 4),
        "true_category": None,
        "equipment_id": [random.choice(EQUIPMENT_POOL) for _ in range(len(texts))],
        "created_at": datetime.now(timezone.utc)
    })
    return df

# -----------------------------------------------------------------------------
# 3. Database Ingestion
# -----------------------------------------------------------------------------
def insert_tickets_to_db(df: pd.DataFrame) -> List[int]:
    """Batch write ticket dataframe into PostgreSQL database."""
    if not DATABASE_URL:
        print("[WARNING] DATABASE_URL not set. Skipping database write.")
        return []

    print("Connecting to PostgreSQL database...")
    # Normalize DATABASE_URL scheme to ensure psycopg2 compatibility
    db_url = DATABASE_URL
    if db_url.startswith("postgresql+psycopg://"):
        db_url = db_url.replace("postgresql+psycopg://", "postgresql+psycopg2://", 1)
    elif db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)

    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=300)
    
    inserted_ids = []
    insert_sql = text("""
        INSERT INTO tickets (text, predicted_category, confidence, true_category, equipment_id, created_at)
        VALUES (:text, :predicted_category, :confidence, :true_category, :equipment_id, :created_at)
        RETURNING id;
    """)

    with engine.begin() as conn:
        for row in df.to_dict(orient="records"):
            res = conn.execute(insert_sql, row)
            inserted_id = res.fetchone()[0]
            inserted_ids.append(inserted_id)

    print(f"Successfully inserted {len(inserted_ids)} records into Postgres `tickets` table.")
    return inserted_ids

# -----------------------------------------------------------------------------
# 4. Low-Confidence Alerting (GitHub Issue Creation)
# -----------------------------------------------------------------------------
def create_github_issue_alert(flagged_df: pd.DataFrame):
    """Create a GitHub Issue alert summarizing low-confidence predictions."""
    if flagged_df.empty:
        print("No low-confidence predictions flagged. No GitHub Issue required.")
        return

    print(f"\n[ALERT] Found {len(flagged_df)} low-confidence predictions requiring human triage!")

    issue_title = f"⚠️ [Automated Alert] {len(flagged_df)} Low-Confidence Maintenance Tickets Flagged (<70%)"
    
    table_rows = []
    for idx, r in flagged_df.iterrows():
        dt_str = str(r["created_at"])[:16]
        table_rows.append(
            f"| `{r.get('id', 'N/A')}` | `{r['equipment_id']}` | **{r['predicted_category']}** | `{r['confidence']:.4f}` ({r['confidence']*100:.1f}%) | `{dt_str}` | {r['text']} |"
        )
    
    table_str = "\n".join(table_rows)

    issue_body = f"""## ⚠️ Automated Low-Confidence Ticket Triage Alert

The scheduled batch ticket ingestion process identified **{len(flagged_df)} ticket(s)** with prediction confidence scores below the **0.70 (70%)** threshold.

These tickets have been stored in the PostgreSQL database with `true_category = NULL` and require human technician verification.

### Flagged Ticket Queue Summary

| Ticket ID | Equipment | Predicted Category | Confidence | Timestamp | Log Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
{table_str}

---
*Automated alert generated by `simulate_ingestion.py` via GitHub Actions scheduled workflow.*
"""

    if GITHUB_TOKEN and GITHUB_REPOSITORY:
        print(f"Creating GitHub Issue on repo `{GITHUB_REPOSITORY}`...")
        url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/issues"
        payload = json.dumps({
            "title": issue_title,
            "body": issue_body,
            "labels": ["automated-alert", "triage-required"]
        }).encode("utf-8")
        
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "Scheduled-Ingestion-Workflow"
            }
        )
        try:
            with urllib.request.urlopen(req) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                print(f"[SUCCESS] Created GitHub Issue #{res_data.get('number')}: {res_data.get('html_url')}")
        except Exception as e:
            print(f"[WARNING] Failed to create GitHub Issue via API: {e}")
    else:
        print("\n--- GITHUB ISSUE BODY (Simulated Local Output) ---")
        print(issue_body)

# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main():
    print("=" * 75)
    print("STAGE 9: SCHEDULED TICKET INGESTION & ALERTING SIMULATION")
    print("=" * 75)

    # 1. Generate Batch
    raw_texts = generate_synthetic_tickets(count=12)

    # 2. Classify
    df_classified = classify_tickets(raw_texts)

    # 3. Database Ingestion
    inserted_ids = insert_tickets_to_db(df_classified)
    if inserted_ids:
        df_classified["id"] = inserted_ids

    # Print Ingested Batch Summary
    print("\nBatch Ingestion Results:")
    for _, r in df_classified.iterrows():
        id_str = f"[ID {r.get('id', 'N/A')}]"
        flag = " [FLAGGED]" if r["confidence"] < 0.70 else ""
        print(f"  {id_str:<10} {r['equipment_id']} | Pred: {r['predicted_category']:<22} | Conf: {r['confidence']:.4f}{flag}")
        print(f"             Text: {r['text'][:85]}...")

    # 4. Low-Confidence Alerting
    flagged_df = df_classified[df_classified["confidence"] < 0.70].copy()
    create_github_issue_alert(flagged_df)

    print("\n[SUCCESS] Ticket ingestion simulation complete!")

if __name__ == "__main__":
    main()
