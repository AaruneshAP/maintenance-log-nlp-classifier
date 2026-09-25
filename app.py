"""
app.py
======
NLP Maintenance Log Classification & Analytics Dashboard (Project 3)

Features:
- Tab 1: Live Analytics Dashboard (PostgreSQL dataset analysis, weekly volume trends,
         category distributions, low-confidence review queue, date range filter)
- Tab 2: Live Model Classifier Demo (Interactive text classification, confidence scoring,
         human-in-the-loop low-confidence flagging indicator)
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NLP Maintenance Log Classifier & Analytics",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished aesthetics and seamless dark/light mode contrast
st.markdown("""
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #FFFFFF !important;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            font-weight: 400;
            color: #B0B0B0 !important;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: var(--secondary-background-color, #F8FAFC);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-color, #0F172A);
        }
        .metric-value-category {
            color: #2563EB;
        }
        [data-theme="dark"] .metric-value-category {
            color: #60A5FA !important;
        }
        .metric-value-confidence {
            color: #059669;
        }
        [data-theme="dark"] .metric-value-confidence {
            color: #34D399 !important;
        }
        .metric-label {
            font-size: 0.85rem;
            color: var(--text-color, #64748B);
            opacity: 0.75;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .flag-badge-danger {
            background-color: rgba(239, 68, 68, 0.18);
            color: #EF4444;
            border: 1px solid rgba(239, 68, 68, 0.35);
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            font-weight: 600;
            display: inline-block;
        }
        .flag-badge-success {
            background-color: rgba(16, 185, 129, 0.18);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.35);
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            font-weight: 600;
            display: inline-block;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Database Connection & Cached Queries
# -----------------------------------------------------------------------------
def get_db_url() -> str:
    """Retrieve DATABASE_URL trying st.secrets first (Streamlit Cloud), falling back to os.getenv/.env."""
    url = None
    try:
        if "DATABASE_URL" in st.secrets:
            url = st.secrets["DATABASE_URL"]
    except Exception:
        pass
    if not url:
        load_dotenv()
        url = os.getenv("DATABASE_URL")
    if url and url.startswith("postgresql+psycopg://"):
        url = url.replace("postgresql+psycopg://", "postgresql+psycopg2://", 1)
    elif url and url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url and url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url

@st.cache_resource
def get_db_engine():
    db_url = get_db_url()
    if not db_url:
        st.error("DATABASE_URL not found in st.secrets, environment variables, or .env file.")
        st.stop()
    return create_engine(db_url, pool_pre_ping=True, pool_recycle=300)

@st.cache_data(ttl=600)
def fetch_all_tickets():
    """Fetch raw tickets table from PostgreSQL."""
    engine = get_db_engine()
    query = """
        SELECT
            id,
            text,
            predicted_category,
            confidence,
            true_category,
            equipment_id,
            created_at
        FROM tickets
        ORDER BY created_at ASC;
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df

@st.cache_data(ttl=600)
def fetch_weekly_category_trends():
    """Execute Query 1 from queries.sql: weekly category frequency."""
    engine = get_db_engine()
    query = """
        SELECT
            DATE_TRUNC('week', created_at)::DATE AS week_start,
            predicted_category,
            COUNT(*) AS ticket_count
        FROM tickets
        GROUP BY week_start, predicted_category
        ORDER BY week_start ASC, predicted_category ASC;
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df

# -----------------------------------------------------------------------------
# Model Artifacts Loader (Tab 2)
# -----------------------------------------------------------------------------
MODELS_DIR = Path("models")

@st.cache_resource
def load_ml_artifacts():
    """Load model, vectorizer, and label encoder."""
    model_path = MODELS_DIR / "best_model.joblib"
    vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    enc_path = MODELS_DIR / "label_encoder.joblib"
    
    if not (model_path.exists() and vec_path.exists() and enc_path.exists()):
        st.error("Model artifacts missing in `models/` directory. Run `train_classifier.py` first.")
        st.stop()
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    encoder = joblib.load(enc_path)
    return model, vectorizer, encoder

def clean_input_text(text_val: str) -> str:
    """Standard text cleaning matching training pipeline."""
    if not isinstance(text_val, str):
        return ""
    t = text_val.lower()
    t = re.sub(r"[^\w\s-]", " ", t)
    t = re.sub(r"\s-\s", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# -----------------------------------------------------------------------------
# Main Application Structure
# -----------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-header" style="color: #FFFFFF !important; font-weight: 700; font-size: 2.2rem; margin-bottom: 0.2rem;">🛠️ NLP Maintenance Log Classifier & Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header" style="color: #B0B0B0 !important; font-size: 1.05rem; margin-bottom: 1.5rem;">Automated failure categorization, SQL database analytics, and real-time technician log triage.</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📊 Analytics Dashboard", "⚡ Live Classifier Demo"])

    # =========================================================================
    # TAB 1: ANALYTICS DASHBOARD
    # =========================================================================
    with tab1:
        st.markdown("### PostgreSQL Database Analytics")
        
        try:
            df_tickets = fetch_all_tickets()
        except Exception as e:
            st.error(f"Failed to fetch data from database: {e}")
            st.stop()

        # Date Range Filter setup
        min_date = df_tickets["created_at"].min().date()
        max_date = df_tickets["created_at"].max().date()

        col_filter1, col_filter2 = st.columns([2, 2])
        with col_filter1:
            date_range = st.date_input(
                "📅 Filter Date Range:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date

        # Filter dataset by selected date range
        mask = (df_tickets["created_at"].dt.date >= start_date) & (df_tickets["created_at"].dt.date <= end_date)
        filtered_df = df_tickets[mask].copy()

        with col_filter2:
            all_cats = sorted(df_tickets["predicted_category"].dropna().unique().tolist())
            selected_cats = st.multiselect(
                "🏷️ Filter Failure Categories:",
                options=all_cats,
                default=all_cats
            )
        
        if selected_cats:
            filtered_df = filtered_df[filtered_df["predicted_category"].isin(selected_cats)]

        st.markdown("---")

        # Top KPI Summary Cards
        total_tickets = len(filtered_df)
        low_conf_tickets = (filtered_df["confidence"] < 0.70).sum()
        low_conf_pct = (low_conf_tickets / total_tickets * 100) if total_tickets > 0 else 0
        top_category = filtered_df["predicted_category"].mode()[0] if not filtered_df.empty else "N/A"
        overall_acc = (filtered_df["predicted_category"] == filtered_df["true_category"]).mean() * 100 if total_tickets > 0 else 0

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Tickets", f"{total_tickets:,}")
        with kpi2:
            st.metric("Low Confidence (<70%)", f"{low_conf_tickets:,}", f"{low_conf_pct:.1f}% flagged", delta_color="inverse")
        with kpi3:
            st.metric("Top Failure Category", top_category.replace('_', ' ').title())
        with kpi4:
            st.metric("Pipeline Integrity Check", f"{overall_acc:.1f}%")
            st.caption("Validates data loading, not model accuracy — see model_choice_rationale.md for true test performance (94.6% F1)")

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row (Line Chart & Bar Chart)
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("📈 Ticket Volume per Category Over Time (Weekly)")
            if not filtered_df.empty:
                # Group by week and category matching Query 1 logic
                weekly_df = (
                    filtered_df.set_index("created_at")
                    .groupby([pd.Grouper(freq="W-MON"), "predicted_category"])
                    .size()
                    .reset_index(name="ticket_count")
                )
                
                fig_line = px.line(
                    weekly_df,
                    x="created_at",
                    y="ticket_count",
                    color="predicted_category",
                    markers=True,
                    labels={"created_at": "Week Start", "ticket_count": "Ticket Count", "predicted_category": "Category"},
                    template="plotly_white"
                )
                fig_line.update_layout(
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    ),
                    margin=dict(l=20, r=20, t=30, b=40),
                    height=380
                )
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("No tickets found for selected filters.")

        with chart_col2:
            st.subheader("📊 Category Distribution")
            if not filtered_df.empty:
                cat_counts = filtered_df["predicted_category"].value_counts().reset_index()
                cat_counts.columns = ["predicted_category", "count"]
                
                fig_bar = px.bar(
                    cat_counts,
                    x="predicted_category",
                    y="count",
                    color="predicted_category",
                    text="count",
                    labels={"predicted_category": "Category", "count": "Tickets"},
                    template="plotly_white"
                )
                fig_bar.update_traces(textposition="auto")
                fig_bar.update_layout(
                    showlegend=False,
                    xaxis_tickangle=-30,
                    margin=dict(l=20, r=20, t=30, b=80),
                    height=380
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No tickets found for selected filters.")

        st.markdown("---")

        # Low-Confidence Review Queue Section
        st.subheader("⚠️ Low-Confidence Review Queue (Confidence < 0.70)")
        st.caption("Tickets with prediction confidence under 70% requiring manual technician review.")

        queue_df = filtered_df[filtered_df["confidence"] < 0.70].copy()

        if not queue_df.empty:
            queue_df = queue_df.sort_values(by="confidence", ascending=True)

            col_q1, col_q2 = st.columns([3, 1])
            with col_q1:
                search_term = st.text_input("🔍 Search ticket text in queue:", placeholder="Type to filter text...")
            with col_q2:
                st.write("")
                st.write("")
                st.info(f"Showing **{len(queue_df)}** flagged tickets")

            if search_term:
                queue_df = queue_df[queue_df["text"].str.contains(search_term, case=False, na=False)]

            display_table = queue_df[[
                "id", "confidence", "predicted_category", "equipment_id", "created_at", "text"
            ]].rename(columns={
                "id": "Ticket ID",
                "confidence": "Confidence",
                "predicted_category": "Predicted Category",
                "equipment_id": "Equipment ID",
                "created_at": "Timestamp",
                "text": "Maintenance Log Description"
            })

            display_table["Confidence"] = display_table["Confidence"].apply(lambda c: f"{c:.4f} ({c*100:.1f}%)")
            display_table["Timestamp"] = display_table["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")

            st.dataframe(
                display_table,
                use_container_width=True,
                height=350,
                hide_index=True
            )
        else:
            st.success("No low-confidence tickets found for the selected date range and categories!")

    # =========================================================================
    # TAB 2: LIVE CLASSIFIER DEMO
    # =========================================================================
    with tab2:
        st.markdown("### Interactive Model Prediction")
        st.write("Type or paste a raw maintenance log description below to test real-time classification and confidence scoring.")

        model, vectorizer, encoder = load_ml_artifacts()

        # Sample Presets Dropdown
        PRESET_SAMPLES = {
            "-- Select a sample preset (or type custom text below) --": "",
            "⚠️ Overheating vs Bearing (Ambiguous)": "Pump outboard bearing temperature elevated to 185F, motor overheating warning on panel.",
            "🔧 Bearing Noise & Vibration (Bearing Failure)": "Primary exhaust fan motor outboard bearing running hot at 195°F with high vibration and loud grinding noise.",
            "💧 High Pressure Hydraulic Leak (Hydraulic Leak)": "Hydraulic press main cylinder seal blown, fluid leaking rapidly onto shop floor under high pressure.",
            "⚡ Overcurrent Fault (Electrical Fault)": "Main circuit breaker tripped on overcurrent fault, control panel displays ground fault alarm.",
        }

        selected_preset = st.selectbox(
            "💡 Load Sample Maintenance Ticket Preset:",
            options=list(PRESET_SAMPLES.keys())
        )
        sample_text = PRESET_SAMPLES[selected_preset]

        with st.form(key="prediction_form"):
            user_input = st.text_area(
                "Maintenance Ticket Description:",
                value=sample_text,
                height=130,
                placeholder="Enter technician log text, e.g., 'Pump 2 mechanical seal leaking cooling water into oil reservoir...'"
            )
            submit_btn = st.form_submit_button("🔍 Classify Ticket", type="primary", use_container_width=True)

        if submit_btn or user_input:
            if not user_input.strip():
                st.warning("Please enter a ticket description to classify.")
            else:
                cleaned = clean_input_text(user_input)
                X_vec = vectorizer.transform([cleaned])
                
                probs = model.predict_proba(X_vec)[0]
                pred_idx = np.argmax(probs)
                predicted_cat = encoder.classes_[pred_idx]
                confidence_score = float(probs[pred_idx])
                confidence_pct = confidence_score * 100.0

                st.markdown("---")
                st.markdown("#### Classification Result")

                res_col1, res_col2, res_col3 = st.columns([2, 2, 2])
                with res_col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Predicted Category</div>
                            <div class="metric-value metric-value-category">{predicted_cat.replace('_', ' ').title()}</div>
                        </div>
                    """, unsafe_allow_html=True)
                with res_col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Model Confidence</div>
                            <div class="metric-value metric-value-confidence">{confidence_pct:.2f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                with res_col3:
                    st.markdown('<div class="metric-card"><div class="metric-label">Triage Status</div><div style="margin-top:0.4rem;">', unsafe_allow_html=True)
                    if confidence_score < 0.70:
                        st.markdown('<span class="flag-badge-danger">⚠️ FLAGGED FOR REVIEW (&lt;70%)</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="flag-badge-success">✅ HIGH CONFIDENCE</span>', unsafe_allow_html=True)
                    st.markdown('</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                if confidence_score < 0.70:
                    st.warning(
                        f"⚠️ **Low Confidence Warning ({confidence_pct:.1f}% < 70.0%)**: "
                        "The model prediction is below the confidence threshold. "
                        "This ticket would be automatically queued in the **Low-Confidence Review Queue** for technician verification."
                    )
                else:
                    st.success(f"✅ **High Confidence ({confidence_pct:.1f}%)**: Automatically classified as `{predicted_cat}`.")

                # Probability Distribution Chart
                st.markdown("##### Failure Category Probability Breakdown")
                prob_df = pd.DataFrame({
                    "Category": [c.replace('_', ' ').title() for c in encoder.classes_],
                    "Probability (%)": probs * 100.0
                }).sort_values(by="Probability (%)", ascending=True)

                fig_probs = px.bar(
                    prob_df,
                    x="Probability (%)",
                    y="Category",
                    orientation="h",
                    text=prob_df["Probability (%)"].apply(lambda v: f"{v:.1f}%"),
                    color="Probability (%)",
                    color_continuous_scale="Blues",
                    template="plotly_white"
                )
                fig_probs.update_traces(textposition="outside")
                fig_probs.update_layout(
                    showlegend=False,
                    coloraxis_showscale=False,
                    xaxis=dict(range=[0, 105]),
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=300
                )
                st.plotly_chart(fig_probs, use_container_width=True)

if __name__ == "__main__":
    main()
