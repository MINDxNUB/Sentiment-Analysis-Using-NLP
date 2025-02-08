from flask import Flask, request, render_template
import pickle
import re
import nltk
import numpy as np
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
import os

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Load model and tokenizer
model = load_model("sentiment_model.h5")

with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Text Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def predict_sentiment(review):
    review = clean_text(review)
    review_seq = tokenizer.texts_to_sequences([review])
    review_padded = pad_sequences(review_seq, maxlen=200)
    prediction = model.predict(review_padded)[0][0]  # Get probability
    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
    return sentiment, prediction

# Flask App
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    probability = None
    wordcloud_path = None

    if request.method == "POST":
        review = request.form["review"]
        sentiment, probability = predict_sentiment(review)

    return render_template("index.html", sentiment=sentiment, probability=probability, wordcloud_path=wordcloud_path)

if __name__ == "__main__":
    app.run(debug=True)