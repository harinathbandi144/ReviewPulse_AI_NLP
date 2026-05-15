import streamlit as st
import joblib
import re
import pandas as pd
import plotly.express as px
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Product Review Sentiment AI",
    page_icon="🛍️",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_files():
    model = joblib.load("sentiment_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

model, tfidf = load_files()

# =========================
# SESSION STATE
# =========================

if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# TEXT CLEANING
# =========================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# PREDICTION FUNCTION
# =========================

def predict_sentiment(review):
    cleaned = clean_text(review)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        confidence = round(max(probabilities) * 100, 2)

    return prediction, confidence, cleaned

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff 0%, #fdf2f8 45%, #ecfeff 100%);
}

.main-card {
    background: rgba(255,255,255,0.88);
    padding: 28px;
    border-radius: 24px;
    box-shadow: 0 20px 50px rgba(15, 23, 42, 0.10);
    border: 1px solid rgba(255,255,255,0.7);
}

.hero-title {
    font-size: 48px;
    font-weight: 900;
    color: #0f172a;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 18px;
    color: #475569;
    margin-top: 10px;
}

.result-positive {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white;
    padding: 28px;
    border-radius: 24px;
    text-align: center;
    font-size: 32px;
    font-weight: 900;
    box-shadow: 0 18px 35px rgba(22,163,74,0.28);
}

.result-negative {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    padding: 28px;
    border-radius: 24px;
    text-align: center;
    font-size: 32px;
    font-weight: 900;
    box-shadow: 0 18px 35px rgba(220,38,38,0.28);
}

.result-neutral {
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white;
    padding: 28px;
    border-radius: 24px;
    text-align: center;
    font-size: 32px;
    font-weight: 900;
    box-shadow: 0 18px 35px rgba(217,119,6,0.28);
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 12px 28px rgba(15,23,42,0.08);
}

.metric-value {
    font-size: 34px;
    font-weight: 900;
    color: #2563eb;
}

.metric-label {
    color: #64748b;
    font-size: 14px;
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 35px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("🛍️ Sentiment AI")

st.sidebar.markdown("""
### Dynamic Features
- Real-time sentiment prediction
- Review history
- Dynamic charts
- Downloadable results
- Confidence score when model supports it
""")

sample_review = st.sidebar.selectbox(
    "Try sample review",
    [
        "",
        "Excellent phone. Battery life is amazing and performance is very smooth.",
        "Worst product. Camera quality is poor and battery drains very fast.",
        "Product is okay. Display is good but camera could be better.",
        "Value for money product with decent performance."
    ]
)

if st.sidebar.button("Clear History"):
    st.session_state.history = []
    st.sidebar.success("History cleared")

# =========================
# HERO SECTION
# =========================

st.markdown("""
<div class="main-card">
    <div class="hero-title">Product Review Sentiment Analyzer</div>
    <div class="hero-subtitle">
        AI-powered ecommerce review analysis using TF-IDF and Machine Learning.
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# =========================
# INPUT SECTION
# =========================

left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    st.subheader("Analyze Customer Review")

    review = st.text_area(
        "Enter product review",
        value=sample_review,
        height=180,
        placeholder="Example: The product quality is excellent, battery backup is strong, and camera is decent..."
    )

    analyze = st.button("🚀 Analyze Sentiment", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    total_reviews = len(st.session_state.history)

    positive_count = sum(1 for item in st.session_state.history if item["Sentiment"] == "Positive")
    negative_count = sum(1 for item in st.session_state.history if item["Sentiment"] == "Negative")
    neutral_count = sum(1 for item in st.session_state.history if item["Sentiment"] == "Neutral")

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{total_reviews}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Total Reviews Analyzed</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{positive_count}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Positive Reviews</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# RESULT SECTION
# =========================

if analyze:
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        prediction, confidence, cleaned_review = predict_sentiment(review)

        if prediction == "Positive":
            st.markdown('<div class="result-positive">😊 Positive Sentiment</div>', unsafe_allow_html=True)
        elif prediction == "Negative":
            st.markdown('<div class="result-negative">😞 Negative Sentiment</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-neutral">😐 Neutral Sentiment</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Prediction", prediction)

        with col2:
            st.metric("Confidence", f"{confidence}%" if confidence else "Not available")

        with col3:
            st.metric("Review Words", len(cleaned_review.split()))

        st.session_state.history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Review": review,
            "Cleaned Review": cleaned_review,
            "Sentiment": prediction,
            "Confidence": confidence if confidence else "Not available"
        })

        with st.expander("View cleaned review used by model"):
            st.write(cleaned_review)

# =========================
# HISTORY + CHARTS
# =========================

if len(st.session_state.history) > 0:

    st.write("")
    st.markdown("## Review Analytics Dashboard")

    history_df = pd.DataFrame(st.session_state.history)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        sentiment_counts = history_df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]

        fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Count",
            title="Sentiment Distribution",
            hole=0.45,
            color="Sentiment",
            color_discrete_map={
                "Positive": "#22c55e",
                "Negative": "#ef4444",
                "Neutral": "#f59e0b"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        fig2 = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            title="Sentiment Count",
            color="Sentiment",
            color_discrete_map={
                "Positive": "#22c55e",
                "Negative": "#ef4444",
                "Neutral": "#f59e0b"
            }
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("## Prediction History")
    st.dataframe(history_df, use_container_width=True)

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction History",
        data=csv,
        file_name="sentiment_prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )

# =========================
# FOOTER
# =========================

st.markdown("""
<div class="footer">
    NLP Sentiment Analysis App | Built with Streamlit
</div>
""", unsafe_allow_html=True)