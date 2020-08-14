import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from os import path


def main():
    st.title("Hello world")
    menu = ["Siddhesh", "Tambe", "App"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Siddhesh":
        st.subheader("Siddhesh")
    elif choice == "Tambe":
        st.subheader("Tambe")
    elif choice == "App":
        st.subheader("App")

    input_text = st.text_input("Say something")
    if input_text != "":
        response_text = get_autocomplete(input_text)
        response = st.text_input('You said {}'.format(response_text))


def get_autocomplete(seed_text):
    next_words = 10
    #tokenizer = Tokenizer()
    #data = open('S:\\AI\\Musical_data_reviewtextonly_file.csv').read()
    #corpus = data.lower().split("\n")[:300]
    #tokenizer.fit_on_texts(corpus)
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    token_list = tokenizer.texts_to_sequences(seed_text)
    input_sequences = []
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)
    max_sequence_len = 819
    model = load_model('autocompletereviewmodel300l10e.h5')
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


if __name__ == '__main__':
    main()
