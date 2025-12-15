import streamlit as st
import joblib
import re
import nltk
import requests
from nltk.corpus import stopwords

nltk.download('stopwords')

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("outputs/model.joblib")
vectorizer = joblib.load("outputs/vectorizer.joblib")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -----------------------------
# News API
# -----------------------------
API_KEY = "9c2f6ba398d74cd9947f31aaf13c2ca2"

def fetch_news(query):
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={query}&language=en&sortBy=relevancy&apiKey={API_KEY}"
    )
    response = requests.get(url)
    data = response.json()

    if data["status"] != "ok" or len(data["articles"]) == 0:
        return None

    article = data["articles"][0]
    return article["title"] + " " + str(article["description"])

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Predict whether a news article is **Fake** or **Real** using ML.")

# Manual input
st.subheader("✍️ Manual News Check")
user_text = st.text_area("Enter news text")

if st.button("Predict Manual News"):
    if user_text.strip() == "":
        st.warning("Please enter news text.")
    else:
        cleaned = clean_text(user_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")

# API input
st.subheader("🌐 Real-Time News (API)")
query = st.text_input("Enter topic (e.g., election, government, economy)")

if st.button("Fetch & Predict News"):
    article = fetch_news(query)

    if article is None:
        st.warning("No news found.")
    else:
        st.write("📰 **Fetched Article:**")
        st.write(article)

        cleaned = clean_text(article)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")
