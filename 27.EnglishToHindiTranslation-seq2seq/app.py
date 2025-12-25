#hindi tokenizer
#english tokenizer
#encoder_model
#decoder_model

from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import streamlit as st

@st.cache_resource
def load_models():
    encoder = load_model('encoder_inference.h5')
    decoder = load_model('decoder_inference.h5')
    encoder_attention=load_model('Encoder_inference_withAttention.h5')
    decoder_attention=load_model('Decoder_inference_withAttention.h5')

    with open('Englishtokenizer.pkl','rb') as f:
        en_tokenizer = pickle.load(f)
    with open('Hinditokenizer.pkl','rb') as f:
        hi_tokenizer = pickle.load(f)

    return encoder, decoder, en_tokenizer, hi_tokenizer,encoder_attention,decoder_attention

encoder_model, decoder_model, tokenizer, tokenizer_hi,encoder_attention,decoder_attention = load_models()


def decode_sequence(input_seq):
    states = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_hi.word_index['<sos>']

    decoded_sentence = []

    while True:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states
        )

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
            break
        sampled_word = tokenizer_hi.index_word.get(sampled_token_index, '')

        if sampled_word == '<eos>' or len(decoded_sentence) > 39:
            break

        decoded_sentence.append(sampled_word)

        target_seq[0, 0] = sampled_token_index
        states = [h, c]

    return ' '.join(decoded_sentence)

def decode_sequence_attention(input_seq):
    encoder_outs, h, c = encoder_attention.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_hi.word_index['<sos>']

    decoded_sentence = []

    for _ in range(40):
        output_tokens, h, c = decoder_attention.predict(
            [target_seq, encoder_outs, h, c],
            verbose=0
        )

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
            break
        sampled_word = tokenizer_hi.index_word[sampled_token_index]

        if sampled_word == '<eos>':
            break

        decoded_sentence.append(sampled_word)
        target_seq[0, 0] = sampled_token_index

    return ' '.join(decoded_sentence)



def clean_sentence(s):
    s = s.lower().strip()
    s=tokenizer.texts_to_sequences([s])
    s=pad_sequences(s,maxlen=30,padding="post")
    return s

st.set_page_config(
    page_title="English → Hindi Translator",
    page_icon="🌐",
    layout="centered"
)

st.title("🌐 English to Hindi Translator")
st.markdown("Translate English sentences into Hindi using a Seq2Seq LSTM model.")

input_text = st.text_area(
    "Enter English sentence",
    height=120,
    placeholder="Example: I am happy today"
)

translate_btn = st.button("🔁 Translate")

if translate_btn:
    if not input_text.strip():
        st.warning("Please enter a sentence to translate.")
    else:
        with st.spinner("Translating..."):
            cleaned = clean_sentence(input_text)
            output = decode_sequence(cleaned)
            output_with_attention=decode_sequence_attention(cleaned)

        st.success("Translation Complete")
        st.text_area(
            "Hindi Translation (without attention)",
            value=output,
            height=120
        )
        st.text_area(
            "Hindi Translation (with attention)",
            value=output_with_attention,
            height=120
        )

st.markdown("---")
st.caption("Seq2Seq LSTM based English → Hindi Translator")
