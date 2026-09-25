"""
main.py
=======
FastAPI Serving Layer for NLP Maintenance Log Classification (Project 3)

Endpoints:
- POST /predict : Accepts technician log text, returns predicted failure category,
                  confidence score, probability breakdown, and logs prediction to Postgres DB.
- GET  /health  : Health check status endpoint.
"""

import os
import re
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

# -----------------------------------------------------------------------------
# Configuration & Environment
# -----------------------------------------------------------------------------
load_dotenv()
MODELS_DIR = Path("models")

def get_db_url() -> Optional[str]:
    """Retrieve DATABASE_URL trying st.secrets first, then os.getenv/.env."""
    url = None
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
            url = st.secrets["DATABASE_URL"]
    except Exception:
        pass
    if not url:
        url = os.getenv("DATABASE_URL")
    if url and url.startswith("postgresql+psycopg://"):
        url = url.replace("postgresql+psycopg://", "postgresql+psycopg2://", 1)
    elif url and url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    return url

# Global ML artifact references
model = None
vectorizer = None
encoder = None
db_engine = None

# -----------------------------------------------------------------------------
# Text Preprocessing Helper
# -----------------------------------------------------------------------------
def clean_text(text_val: str) -> str:
    """Clean text matching classifier training pipeline."""
    if not isinstance(text_val, str):
        return ""
    t = text_val.lower()
    t = re.sub(r"[^\w\s-]", " ", t)
    t = re.sub(r"\s-\s", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# -----------------------------------------------------------------------------
# FastAPI Lifespan Context (Startup & Shutdown)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vectorizer, encoder, db_engine
    
    # 1. Load ML Artifacts
    model_path = MODELS_DIR / "best_model.joblib"
    vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    enc_path = MODELS_DIR / "label_encoder.joblib"
    
    if not (model_path.exists() and vec_path.exists() and enc_path.exists()):
        raise RuntimeError("ML model artifacts missing in `models/` directory. Run `train_classifier.py` first.")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    encoder = joblib.load(enc_path)
    print("ML model artifacts loaded successfully.")

    # 2. Initialize Database Connection Engine
    db_url = get_db_url()
    if db_url:
        try:
            db_engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=300)
            with db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("PostgreSQL database connection initialized successfully.")
        except Exception as e:
            print(f"Warning: Database connection failed ({e}). Running without DB logging.")
            db_engine = None
    else:
        print("Warning: DATABASE_URL not found. Running without DB logging.")

    yield
    
    # Cleanup on shutdown
    if db_engine:
        db_engine.dispose()
        print("Database connections closed.")

# -----------------------------------------------------------------------------
# FastAPI App Initialization
# -----------------------------------------------------------------------------
app = FastAPI(
    title="NLP Maintenance Log Classification API",
    description="Production API endpoint serving real-time failure category predictions for technician maintenance logs with PostgreSQL logging.",
    version="1.0.0",
    lifespan=lifespan
)

# -----------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        description="Free-text technician maintenance log description",
        example="Primary cooling pump outboard bearing housing running hot at 195°F with high vibration."
    )
    equipment_id: Optional[str] = Field(
        default="API-101",
        description="Synthetic or real equipment identifier",
        example="EQ-105"
    )

class PredictResponse(BaseModel):
    predicted_category: str = Field(..., description="Top predicted failure category")
    confidence: float = Field(..., description="Model probability score (0.0 to 1.0)")
    confidence_pct: str = Field(..., description="Formatted confidence percentage string")
    flagged_for_review: bool = Field(..., description="True if confidence < 0.70 (requires manual review)")
    probabilities: Dict[str, float] = Field(..., description="Full probability breakdown across all 8 categories")
    db_record_id: Optional[int] = Field(default=None, description="Primary key ID of created tickets DB row")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
    timestamp: str

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Service health check verifying model status and DB connectivity."""
    db_connected = False
    if db_engine:
        try:
            with db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_connected = True
        except Exception:
            db_connected = False

    return HealthResponse(
        status="healthy" if (model is not None) else "unhealthy",
        model_loaded=model is not None,
        database_connected=db_connected,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK, tags=["Predictions"])
def predict(request: PredictRequest):
    """
    Classify a free-text maintenance ticket log into failure categories.
    Returns prediction, confidence score, full probability breakdown, and logs record to database.
    """
    if not model or not vectorizer or not encoder:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service unavailable or not fully initialized."
        )

    raw_text = request.text.strip()
    
    # Input Validation: Reject empty or whitespace-only text
    if not raw_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text cannot be empty or whitespace-only."
        )
    
    # Input Validation: Reject excessively short or long text
    if len(raw_text) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text too short. Please provide a descriptive maintenance log (min 3 characters)."
        )
    
    if len(raw_text) > 2000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text exceeds maximum allowed length of 2000 characters."
        )

    # 1. Clean & Vectorize Text
    cleaned = clean_text(raw_text)
    X_vec = vectorizer.transform([cleaned])

    # 2. Model Prediction & Confidence
    probs = model.predict_proba(X_vec)[0]
    pred_idx = int(np.argmax(probs))
    predicted_category = str(encoder.classes_[pred_idx])
    confidence = float(np.round(probs[pred_idx], 4))
    flagged = confidence < 0.70

    # Build full probability dictionary sorted by probability descending
    prob_dict = {
        str(cat): float(np.round(p, 4))
        for cat, p in sorted(
            zip(encoder.classes_, probs),
            key=lambda x: x[1],
            reverse=True
        )
    }

    # 3. Log Prediction to PostgreSQL DB
    db_record_id = None
    if db_engine:
        try:
            insert_sql = text("""
                INSERT INTO tickets (text, predicted_category, confidence, equipment_id, created_at)
                VALUES (:text, :predicted_category, :confidence, :equipment_id, :created_at)
                RETURNING id;
            """)
            with db_engine.begin() as conn:
                res = conn.execute(insert_sql, {
                    "text": raw_text,
                    "predicted_category": predicted_category,
                    "confidence": confidence,
                    "equipment_id": request.equipment_id or "API-101",
                    "created_at": datetime.now(timezone.utc)
                })
                row = res.fetchone()
                if row:
                    db_record_id = row[0]
        except Exception as e:
            print(f"Warning: Failed to log prediction to PostgreSQL database: {e}")

    return PredictResponse(
        predicted_category=predicted_category,
        confidence=confidence,
        confidence_pct=f"{confidence * 100:.2f}%",
        flagged_for_review=flagged,
        probabilities=prob_dict,
        db_record_id=db_record_id
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
