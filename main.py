import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential,load_model

#load the imdb model
word_index = imdb.get_word_index()
reversed_index = {value:key for key, value in word_index.items()}

#load Model
model = load_model('simple_rnn_imdb.h5')

# Create a function
def decode_review(review):
    return ' '.join([reversed_index.get(word - 3, '?') for word in review])

def preprocess_text(text):
    words = text.lower().split()
    encode_review = ([word_index.get(word,2)+3 for word in words])
    padded_review = sequence.pad_sequences([encode_review], maxlen = 500)
    return padded_review

def prediction_review(text):
    preprocessed_text = preprocess_text(text)
    prediction = model.predict(preprocessed_text)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment,prediction[0][0]

st.title("Sentiment Analysis")

user_input=st.text_area("Enter a content")

if st.button('Classify'):
    sentiment,score = prediction_review(user_input)

    st.write(f'Sentiment : {sentiment}')
    st.write(f'prediction Score : {score}')

else:
    st.write('Please enter content')