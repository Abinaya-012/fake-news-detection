import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')

# Load data
fake = pd.read_csv("../data/Fake.csv")
true = pd.read_csv("../data/True.csv")

fake['label'] = 0
true['label'] = 1
df = pd.concat([fake, true])[['text', 'label']]

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['text'] = df['text'].apply(clean_text)

X = joblib.load("../outputs/vectorizer.joblib").transform(df['text'])
y = df['label']

model = joblib.load("../outputs/model.joblib")
y_pred = model.predict(X)

print("Accuracy:", accuracy_score(y, y_pred))
print(classification_report(y, y_pred))
