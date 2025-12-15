import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load saved model and vectorizer
model = joblib.load("../outputs/model.joblib")
vectorizer = joblib.load("../outputs/vectorizer.joblib")

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Take input from user
print("\n📰 Fake News Detection System")
news = input("Enter news text:\n")

cleaned_news = clean_text(news)
vectorized_news = vectorizer.transform([cleaned_news])

prediction = model.predict(vectorized_news)

if prediction[0] == 1:
    print("\n✅ Prediction: REAL NEWS")
else:
    print("\n❌ Prediction: FAKE NEWS")
