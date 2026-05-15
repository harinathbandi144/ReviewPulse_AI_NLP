import streamlit as st
import joblib
import re
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="ReviewPulse AI",
    page_icon="🛒",
    layout="wide"
)

@st.cache_resource
def load_files():
    model = joblib.load("sentiment_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

model, tfidf = load_files()

if "history" not in st.session_state:
    st.session_state.history = []

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict_sentiment(review):
    cleaned = clean_text(review)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        confidence = round(max(probabilities) * 100, 2)

    return prediction, confidence, cleaned

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 20%, rgba(255,153,0,0.28), transparent 30%),
        radial-gradient(circle at 85% 15%, rgba(37,99,235,0.28), transparent 28%),
        radial-gradient(circle at 50% 90%, rgba(236,72,153,0.22), transparent 35%),
        linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
    color: white;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #111827);
    border-right: 1px solid rgba(255,255,255,0.12);
}

.main-glass {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 30px;
    padding: 32px;
    box-shadow: 0 30px 90px rgba(0,0,0,0.35);
    backdrop-filter: blur(18px);
}

.hero {
    background:
        linear-gradient(135deg, rgba(255,153,0,0.25), rgba(37,99,235,0.25)),
        rgba(255,255,255,0.09);
    border-radius: 34px;
    padding: 44px;
    border: 1px solid rgba(255,255,255,0.20);
    box-shadow: 0 35px 100px rgba(0,0,0,0.40);
}

.brand-pill {
    display: inline-block;
    padding: 9px 18px;
    border-radius: 999px;
    background: linear-gradient(90deg, #ff9900, #facc15, #22c55e);
    color: #020617;
    font-size: 13px;
    font-weight: 900;
    letter-spacing: 1px;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 58px;
    font-weight: 950;
    line-height: 1.03;
    background: linear-gradient(90deg, #ffffff, #facc15, #ff9900, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    font-size: 19px;
    color: #dbeafe;
    margin-top: 16px;
    max-width: 850px;
}

.product-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.16), rgba(255,255,255,0.06));
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 28px;
    padding: 28px;
    box-shadow: 0 28px 70px rgba(0,0,0,0.30);
    backdrop-filter: blur(16px);
}

.metric-box {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 24px;
    padding: 24px;
    text-align: center;
}

.metric-num {
    font-size: 38px;
    font-weight: 950;
    background: linear-gradient(90deg, #facc15, #ff9900);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    color: #cbd5e1;
    font-size: 14px;
}

.result-positive {
    background: linear-gradient(135deg, #16a34a, #22c55e, #bbf7d0);
    color: #052e16;
    padding: 34px;
    border-radius: 32px;
    text-align: center;
    font-size: 38px;
    font-weight: 950;
    box-shadow: 0 0 45px rgba(34,197,94,0.55);
}

.result-negative {
    background: linear-gradient(135deg, #991b1b, #ef4444, #fecaca);
    color: white;
    padding: 34px;
    border-radius: 32px;
    text-align: center;
    font-size: 38px;
    font-weight: 950;
    box-shadow: 0 0 45px rgba(239,68,68,0.55);
}

.result-neutral {
    background: linear-gradient(135deg, #92400e, #f59e0b, #fde68a);
    color: #111827;
    padding: 34px;
    border-radius: 32px;
    text-align: center;
    font-size: 38px;
    font-weight: 950;
    box-shadow: 0 0 45px rgba(245,158,11,0.55);
}

.stTextArea textarea {
    background: rgba(255,255,255,0.92);
    color: #020617;
    border-radius: 22px;
    border: 2px solid rgba(255,153,0,0.55);
    font-size: 17px;
}

.stButton button {
    background: linear-gradient(90deg, #ff9900, #facc15, #22c55e);
    color: #020617;
    border: none;
    border-radius: 18px;
    padding: 14px 22px;
    font-weight: 950;
    font-size: 17px;
    box-shadow: 0 0 30px rgba(255,153,0,0.42);
}

.stButton button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 45px rgba(250,204,21,0.65);
}

.footer {
    color: #cbd5e1;
    text-align: center;
    margin-top: 35px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🛒 ReviewPulse AI")
st.sidebar.write("Amazon / Flipkart style review intelligence dashboard.")

sample_review = st.sidebar.selectbox(
    "Try a sample review",
    [
        "",
        "Excellent phone. Battery backup is amazing and performance is very smooth.",
        "Worst product. Camera quality is poor and battery drains very fast.",
        "Product is okay. Display is good but camera could be better.",
        "Value for money product with decent performance."
    ]
)

if st.sidebar.button("Clear Dashboard History"):
    st.session_state.history = []
    st.sidebar.success("Cleared")

st.markdown("""
<div class="hero">
    <div class="brand-pill">ECOMMERCE REVIEW INTELLIGENCE</div>
    <div class="hero-title">Review Sentiment Analyzer</div>
    <div class="hero-sub">
        A cinematic AI dashboard that analyzes product reviews like Amazon or Flipkart review intelligence.
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

col1, col2 = st.columns([2.2, 1])

with col1:
    st.markdown('<div class="product-card">', unsafe_allow_html=True)
    st.subheader("📝 Analyze Product Review")

    review = st.text_area(
        "Paste customer review",
        value=sample_review,
        height=190,
        placeholder="Example: The product quality is excellent, delivery was fast, and battery backup is impressive..."
    )

    analyze = st.button("✨ Analyze Review Sentiment", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    total = len(st.session_state.history)
    positive = sum(1 for x in st.session_state.history if x["Sentiment"] == "Positive")
    negative = sum(1 for x in st.session_state.history if x["Sentiment"] == "Negative")
    neutral = sum(1 for x in st.session_state.history if x["Sentiment"] == "Neutral")

    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-num">{total}</div>
        <div class="metric-label">Total Reviews</div>
    </div>
    <br>
    <div class="metric-box">
        <div class="metric-num">{positive}</div>
        <div class="metric-label">Positive Reviews</div>
    </div>
    <br>
    <div class="metric-box">
        <div class="metric-num">{negative}</div>
        <div class="metric-label">Negative Reviews</div>
    </div>
    """, unsafe_allow_html=True)

if analyze:
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        prediction, confidence, cleaned_review = predict_sentiment(review)

        if prediction == "Positive":
            st.markdown('<div class="result-positive">😊 POSITIVE REVIEW</div>', unsafe_allow_html=True)
        elif prediction == "Negative":
            st.markdown('<div class="result-negative">😞 NEGATIVE REVIEW</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-neutral">😐 NEUTRAL REVIEW</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment", prediction)
        c2.metric("Confidence", f"{confidence}%" if confidence else "Not Available")
        c3.metric("Words", len(cleaned_review.split()))

        st.session_state.history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Review": review,
            "Cleaned Review": cleaned_review,
            "Sentiment": prediction,
            "Confidence": confidence if confidence else "Not Available"
        })

        with st.expander("View cleaned review"):
            st.write(cleaned_review)

if len(st.session_state.history) > 0:
    st.markdown("## 📊 Live Review Analytics")

    history_df = pd.DataFrame(st.session_state.history)

    sentiment_counts = history_df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    chart1, chart2 = st.columns(2)

    with chart1:
        fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Count",
            hole=0.55,
            title="Sentiment Distribution",
            color="Sentiment",
            color_discrete_map={
                "Positive": "#22c55e",
                "Negative": "#ef4444",
                "Neutral": "#f59e0b"
            }
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart2:
        fig2 = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            title="Review Sentiment Count",
            color="Sentiment",
            color_discrete_map={
                "Positive": "#22c55e",
                "Negative": "#ef4444",
                "Neutral": "#f59e0b"
            }
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("## 🧾 Prediction History")
    st.dataframe(history_df, use_container_width=True)

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Review Report",
        csv,
        "sentiment_report.csv",
        "text/csv",
        use_container_width=True
    )

st.markdown("""
<div class="footer">
    ReviewPulse AI | Cinematic Ecommerce Sentiment Dashboard
</div>
""", unsafe_allow_html=True)