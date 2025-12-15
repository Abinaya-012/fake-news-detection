import pandas as pd
import joblib
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Download stopwords once
nltk.download('stopwords')

# -------------------------
# 1. Load Dataset
# -------------------------
fake = pd.read_csv("../data/Fake.csv")
true = pd.read_csv("../data/True.csv")

fake['label'] = 0   # Fake news
true['label'] = 1   # Real news

df = pd.concat([fake, true])
df = df[['text', 'label']]
df.reset_index(drop=True, inplace=True)

# -------------------------
# 2. Text Cleaning Function
# -------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Feature Extraction
if 'title' in df.columns:
    df['text'] = df['title'].fillna('') + " " + df['text']

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)
)

X = tfidf.fit_transform(df['text'])
y = df['label']


# -------------------------
# 4. Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 5. Train Model
# -------------------------
model = LinearSVC()
model.fit(X_train, y_train)

# -------------------------
# 6. Save Model
# -------------------------
joblib.dump(model, "../outputs/model.joblib")
joblib.dump(tfidf, "../outputs/vectorizer.joblib")

print("✅ Model training completed and saved successfully!")
