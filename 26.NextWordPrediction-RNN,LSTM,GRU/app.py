import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model


@st.cache_resource
def load_models():
    model_RNN = load_model("BestRNNModel.h5")
    model_LSTM = load_model("LSTM.h5")
    model_GRU = load_model("GRU.h5")
    return model_RNN, model_LSTM, model_GRU

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

def predictionPipeline(sentence, next_words, model, tokenizer):
    sentence = sentence.lower()
    for i in range(next_words):
        token_list = tokenizer.texts_to_sequences([sentence])[0]
        token_list = pad_sequences([token_list], maxlen=17, padding='pre')
        predicted_probabilities = model.predict(token_list, verbose=0)
        idx = np.argmax(predicted_probabilities)
        predicted_word = tokenizer.index_word.get(idx, "")
        sentence += " " + predicted_word
    return sentence


st.title("Next Word Prediction (RNN / LSTM / GRU)")

text = st.text_input("Enter a sentence")

if not text.strip():
    st.stop()

next_words = st.slider("Number of words to generate", 1, 10, 5)


model_RNN, model_LSTM, model_GRU = load_models()
tokenizer = load_tokenizer()


RNN_output = predictionPipeline(text, next_words, model_RNN, tokenizer)
LSTM_output = predictionPipeline(text, next_words, model_LSTM, tokenizer)
GRU_output = predictionPipeline(text, next_words, model_GRU, tokenizer)


st.subheader("Generated Outputs:")
st.write("**RNN Output:** ", RNN_output)
st.write("**LSTM Output:** ", LSTM_output)
st.write("**GRU Output:** ", GRU_output)
