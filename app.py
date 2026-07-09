import streamlit as st
import pickle
import re
import string
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# --------------------------------------
# Load saved model files
# --------------------------------------
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

stop_words = set(stopwords.words("english"))

def clean(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

# --------------------------------------
# Web page UI
# --------------------------------------
st.set_page_config(page_title="Hate Speech Detector", page_icon="🛡️")

st.title("🛡️ Hate Speech Detector")
st.write("Type a sentence below and the model will classify it.")

user_input = st.text_area("Enter text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        cleaned = clean(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)
        label = encoder.inverse_transform(prediction)[0]

        if label == "Hate Speech":
            st.error(f"Prediction: **{label}**")
        elif label == "Offensive Language":
            st.warning(f"Prediction: **{label}**")
        else:
            st.success(f"Prediction: **{label}**")
