# fake_news_pipeline.py
import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

DATA_DIR = 'data'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

fake_path = os.path.join(DATA_DIR, 'Fake.csv')
true_path = os.path.join(DATA_DIR, 'True.csv')

if not (os.path.exists(fake_path) and os.path.exists(true_path)):
    raise FileNotFoundError('Place Fake.csv and True.csv in the data/ folder')

fake = pd.read_csv(fake_path)
true = pd.read_csv(true_path)

fake['label'] = 0
true['label'] = 1

df = pd.concat([fake, true], ignore_index=True)

text_col = None
for c in ['text','article','content','title']:
    if c in df.columns:
        text_col = c
        break
if text_col is None:
    raise ValueError('No text column found in dataset')

df = df[[text_col,'label']].rename(columns={text_col:'text'})
df = df.dropna(subset=['text']).sample(frac=1, random_state=42).reset_index(drop=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
url_pattern = re.compile(r'https?://\S+|www\.\S+')
non_alpha = re.compile(r'[^a-zA-Z ]')

def clean_text(s):
    s = str(s)
    s = url_pattern.sub('', s)
    s = s.lower()
    s = non_alpha.sub(' ', s)
    tokens = nltk.word_tokenize(s)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)

nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

def evaluate_model(model, X_tfidf, y_true):
    y_pred = model.predict(X_tfidf)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred)
    return {'accuracy':acc, 'precision':prec, 'recall':rec, 'f1':f1, 'cm':cm, 'cr':cr}

metrics_lr = evaluate_model(lr, X_test_tfidf, y_test)
metrics_nb = evaluate_model(nb, X_test_tfidf, y_test)

best_model = lr if metrics_lr['f1'] >= metrics_nb['f1'] else nb
joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.joblib'))
joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))

if isinstance(best_model, LogisticRegression):
    feat_names = vectorizer.get_feature_names_out()
    coef = best_model.coef_[0]
    top_pos_idx = np.argsort(coef)[-20:][::-1]
    top_neg_idx = np.argsort(coef)[:20]
    print('TOP_REAL =', [feat_names[i] for i in top_pos_idx[:20]])
    print('TOP_FAKE =', [feat_names[i] for i in top_neg_idx[:20]])

app_code = """
# app.py
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

model = joblib.load('models/best_model.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

url_pattern = re.compile(r'https?://\\S+|www\\.\\S+')
non_alpha = re.compile(r'[^a-zA-Z ]')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(s):
    s = str(s)
    s = url_pattern.sub('', s)
    s = s.lower()
    s = non_alpha.sub(' ', s)
    tokens = nltk.word_tokenize(s)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

st.title('Fake vs Real News Classifier')
text = st.text_area('Paste news article text or headline here', height=300)
if st.button('Predict'):
    if not text.strip():
        st.warning('Please paste some text.')
    else:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        label = 'Real' if pred == 1 else 'Fake'
        confidence = max(prob)
        st.markdown(f'**Prediction:** {label}  \
**Confidence:** {confidence:.3f}')
        if hasattr(model, 'coef_'):
            fname = vectorizer.get_feature_names_out()
            coefs = model.coef_[0]
            present_idx = [i for i,ft in enumerate(fname) if ft in cleaned.split()]
            present_scores = [(fname[i], coefs[i]) for i in present_idx]
            present_scores = sorted(present_scores, key=lambda x: x[1], reverse=True)[:10]
            if present_scores:
                st.subheader('Top contributing tokens (positive => real, negative => fake)')
                for tok,score in present_scores:
                    st.write(f'{tok}: {score:.4f}')
"""

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

print('Training complete. Models saved to models/. Streamlit app written to app.py')
