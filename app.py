import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Load model and vectorizer
model = joblib.load('models/best_model.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

# Download NLTK data if not already present
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# Streamlit UI
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🧠", layout="centered")

st.title("📰 Fake News Detection System")
st.markdown("### Enter any news headline or short paragraph below to check if it’s **Real** or **Fake.**")

# User input
user_input = st.text_area("📝 Paste your news text here:", height=150)

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text before analyzing.")
    else:
        clean = clean_text(user_input)
        vector = vectorizer.transform([clean])
        prediction = model.predict(vector)[0]

        if prediction == 0:
            st.success("✅ The news appears to be **REAL**.")
        else:
            st.error("🚨 The news appears to be **FAKE**.")
        
        st.caption("Model: Logistic Regression (TF-IDF based)")

st.markdown("---")
st.markdown("👨‍💻 Developed by Lijesh S | Fake News Detection Project")
