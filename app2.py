import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(
    page_title="NLP Sentiment Analasys",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "page" not in st.session_state:
    st.session_state.page = "landing"

def go_to(page_name):
    st.session_state.page = page_name
    st.rerun()

@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_model()
except Exception:
    model, vectorizer = None, None

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;800;900&family=Poppins:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 25%, rgba(0, 200, 255, 0.25), transparent 28%),
        radial-gradient(circle at 85% 15%, rgba(90, 80, 255, 0.30), transparent 30%),
        linear-gradient(135deg, #020617, #07142e, #020617);
    color: white;
}

#MainMenu, header, footer {
    visibility: hidden;
}

.block-container {
    padding-top: 2.5rem;
    padding-left: 4rem;
    padding-right: 4rem;
}

.main-title {
    text-align: center;
    font-family: 'Orbitron', sans-serif;
    font-size: 72px;
    font-weight: 900;
    color: #22d3ee;
    text-shadow: 0 0 25px #22d3ee, 0 0 60px rgba(34,211,238,0.7);
    margin-top: 10px;
    margin-bottom: 10px;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    letter-spacing: 10px;
    color: #dbeafe;
    margin-bottom: 65px;
}

.logo {
    font-family: 'Orbitron', sans-serif;
    font-size: 28px;
    font-weight: 900;
    color: #22d3ee;
    text-shadow: 0 0 22px #22d3ee;
    margin-bottom: 55px;
}

.stButton > button {
    width: 100%;
    height: 68px;
    border-radius: 18px;
    border: 1px solid rgba(125,211,252,0.8);
    background: linear-gradient(90deg, #0284c7, #2563eb, #4f46e5);
    color: white !important;
    font-size: 20px;
    font-weight: 700;
    box-shadow: 0 0 28px rgba(59,130,246,0.55);
    transition: 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 0 45px rgba(34,211,238,1);
    border: 1px solid #22d3ee;
}

textarea {
    background: rgba(51,65,85,0.88) !important;
    border: 2px solid rgba(125,211,252,0.95) !important;
    color: white !important;
    border-radius: 18px !important;
    font-size: 18px !important;
    min-height: 250px !important;
    box-shadow: 0 0 20px rgba(34,211,238,0.25);
}

label {
    color: #dbeafe !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}

.result-box {
    text-align: center;
    padding: 28px;
    border-radius: 22px;
    font-size: 34px;
    font-weight: 900;
    margin-top: 30px;
}

.positive {
    background: linear-gradient(135deg, #15803d, #22c55e);
    box-shadow: 0 0 35px rgba(34,197,94,0.75);
}

.negative {
    background: linear-gradient(135deg, #be123c, #f43f5e);
    box-shadow: 0 0 35px rgba(244,63,94,0.75);
}

.neutral {
    background: linear-gradient(135deg, #ca8a04, #facc15);
    color: #111827;
    box-shadow: 0 0 35px rgba(250,204,21,0.75);
}

.metric-box {
    padding: 25px;
    border-radius: 20px;
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(34,211,238,0.35);
    text-align: center;
    box-shadow: 0 0 25px rgba(34,211,238,0.20);
}

.metric-label {
    color: #94a3b8;
    font-size: 15px;
}

.metric-value {
    color: #22d3ee;
    font-size: 34px;
    font-weight: 900;
}

.page-heading {
    font-size: 42px;
    font-weight: 900;
    color: #f8fafc;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

def sidebar_nav():
    st.markdown('<div class="logo"></div>', unsafe_allow_html=True)

    if st.button("🧠 Test Sentiment"):
        go_to("test")

    st.write("")

    if st.button("📊 Analytics / Graphs"):
        go_to("analytics")

    st.write("")

    if st.button("🤖 Models Built"):
        go_to("models")

    st.write("")

    if st.button("🏠 Home"):
        go_to("landing")

def app_header():
    st.markdown('<div class="main-title">⚡ Review Pulse AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI POWERED SENTIMENT ANALYSIS PLATFORM</div>', unsafe_allow_html=True)

if st.session_state.page == "landing":

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    app_header()

    # c1, c2, c3 = st.columns([1.4, 1, 1.4])
    # with c2:
    #     if st.button("🚀 ENTER DASHBOARD"):
    #         go_to("test")
    st.markdown("<br><br>", unsafe_allow_html=True)

    left, center, right = st.columns([2.2, 1.8, 2.2])

    with center:
        if st.button("🚀 ENTER DASHBOARD"):
            go_to("test")

elif st.session_state.page == "test":

    app_header()

    nav, main = st.columns([1.1, 4], gap="large")

    with nav:
        sidebar_nav()

    with main:
        review = st.text_area(
            "Customer Review",
            height=250,
            placeholder="Example: The product quality is excellent and delivery was very fast..."
        )

        st.write("")

        if st.button("⚡ Predict Sentiment"):
            if review.strip() == "":
                st.warning("Please enter a review first.")
            elif model is None or vectorizer is None:
                st.error("Model files not found. Add sentiment_model.pkl and tfidf_vectorizer.pkl.")
            else:
                review_vec = vectorizer.transform([review])
                prediction = model.predict(review_vec)[0]
                pred = str(prediction).lower()

                if pred == "positive":
                    st.markdown('<div class="result-box positive">😊 POSITIVE SENTIMENT</div>', unsafe_allow_html=True)
                elif pred == "negative":
                    st.markdown('<div class="result-box negative">😞 NEGATIVE SENTIMENT</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-box neutral">😐 NEUTRAL SENTIMENT</div>', unsafe_allow_html=True)

elif st.session_state.page == "analytics":

    app_header()

    nav, main = st.columns([1.1, 4], gap="large")

    with nav:
        sidebar_nav()

    with main:
        st.markdown('<div class="page-heading">Sentiment Analytics</div>', unsafe_allow_html=True)

        data = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Count": [6124, 2845, 1285]
        })

        total = data["Count"].sum()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f'''
            <div class="metric-box">
                <div class="metric-label">Total Reviews</div>
                <div class="metric-value">{total}</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            '''
            <div class="metric-box">
                <div class="metric-label">Positive</div>
                <div class="metric-value">6,124</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            '''
            <div class="metric-box">
                <div class="metric-label">Negative</div>
                <div class="metric-value">2,845</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            '''
            <div class="metric-box">
                <div class="metric-label">Neutral</div>
                <div class="metric-value">1,285</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.write("")
        st.write("")

        g1, g2 = st.columns(2)

        with g1:
            fig1 = px.pie(data, names="Sentiment", values="Count", hole=0.55, title="Sentiment Distribution")
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig1, use_container_width=True)

        with g2:
            fig2 = px.bar(data, x="Sentiment", y="Count", text="Count", title="Sentiment Count")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig2, use_container_width=True)

elif st.session_state.page == "models":

    app_header()

    nav, main = st.columns([1.1, 4], gap="large")

    with nav:
        sidebar_nav()

    with main:
        st.markdown('<div class="page-heading">Models Built</div>', unsafe_allow_html=True)

        model_df = pd.DataFrame({
            "Model": [
                "Linear SVM",
                "Logistic Regression",
                "SGD Classifier",
                "Complement Naive Bayes",
                "Random Forest",
                "Extra Trees",
                "Gradient Boosting"
            ],
            "Feature Engineering": [
                "TF-IDF", "TF-IDF", "TF-IDF", "TF-IDF", "TF-IDF", "TF-IDF", "TF-IDF"
            ],
            "Status": [
                "Best Model", "Tested", "Tested", "Tested", "Tested", "Tested", "Tested"
            ]
        })

        st.dataframe(model_df, use_container_width=True, hide_index=True)