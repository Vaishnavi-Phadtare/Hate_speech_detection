import pandas as pd
import numpy as np
import re
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from nltk.corpus import stopwords
from sklearn.utils import resample

nltk.download('stopwords')

# === Load and Preprocess Dataset ===
file_path = os.path.abspath("Ethos_Dataset_Binary.csv")
data = pd.read_csv(file_path)

# Fix malformed CSV with combined columns
data['comment;isHate'] = data['comment;isHate'].str.replace(',', '.')
valid_rows = data['comment;isHate'].str.count(';') == 1
data = data[valid_rows]
split_data = data['comment;isHate'].str.split(';', expand=True)
split_data.columns = ['comment', 'isHate']
data = pd.concat([split_data, data.drop(columns=['comment;isHate'])], axis=1)
data['isHate'] = pd.to_numeric(data['isHate'], errors='coerce')
data = data.dropna(subset=['isHate'])
data["isHate"] = data["isHate"].astype(int)
data["labels"] = data['isHate'].map({0: "Not-Hate Speech", 1: "Hate Speech"})

# === Clean the Text ===
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()
    text = re.sub('\n', '', text)
    return text

data["comment"] = data["comment"].apply(clean_text)

# === Define x and y before using ===
x = data["comment"].values
y = data["isHate"].values

# === Upsampling minority class ===
df = pd.DataFrame({'comment': x, 'isHate': y})
hate = df[df['isHate'] == 1]
not_hate = df[df['isHate'] == 0]

# Upsample hate class
hate_upsampled = resample(hate, replace=True, n_samples=len(not_hate), random_state=42)

# Combine and shuffle
df_balanced = pd.concat([not_hate, hate_upsampled]).sample(frac=1, random_state=42)

# Redefine x and y after balancing
x = df_balanced['comment'].values
y = df_balanced['isHate'].values.astype(int)

# === Vectorize and Train ===
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
X = vectorizer.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# === Evaluate ===
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Train Performance:\n", classification_report(y_train, y_pred_train))
print("\nTest Performance:\n", classification_report(y_test, y_pred_test))
