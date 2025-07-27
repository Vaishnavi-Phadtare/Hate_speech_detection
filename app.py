import streamlit as st
import joblib
from preprocessing import clean_text


model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("🛡️ Hate Speech Detection App")
st.write("Enter a message to check if it's hate speech.")

text_input = st.text_area("Enter text here:")

if st.button("Predict"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(text_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0][1]

        if prediction == 1:
            st.error(f"Hate Speech ❌ ({prob*100:.2f}% confidence)")
        else:
            st.success(f"Not Hate Speech ✅ ({(1 - prob)*100:.2f}% confidence)")
