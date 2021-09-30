import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

__author__ = "Siddhesh Tambe"


def main():
    st.title("Customer Review Writing Assistant")
    st.sidebar.title("How was it made")
     st.sidebar.markdown("""Made by Siddhesh Tambe""")
    st.sidebar.markdown("""1.This model was trained on a part of the Amazon Product review dataset""")
    st.sidebar.markdown(
        """2.This model uses the capability of NLP to suggest future text based on current user input""")
    st.sidebar.markdown("""3.The model uses a simple feed forward network. The capabilities of the model can be 
    improved with additional training data,hyperparameter tuning and computing power""")
    st.sidebar.markdown("""4.With this, customer can write reviews with suggestions. This would increase number of 
    customer reviews for a product and give an organization an oppotunity to  better their customer service""")
    st.write("This app helps ypu write reviews about music products. This helps companies in increasing customer "
             "engagement with the product and ultimately improve CLV")
    word_number = st.slider('Select number of words', 0, 130, 25)
    input_text = st.text_area("Start writing the review, get suggestions for the next " + str(word_number) + " words")

    if st.button("Get suggestion"):
        if input_text != "":
            response_text = get_autocomplete(input_text, word_number)
            st.text_area("", response_text)


def recursive_call_function(input_text, word_number):
    response_text = get_autocomplete(input_text, word_number)
    st.text_area("", response_text)


def get_autocomplete(seed_text, word_number):
    next_words = word_number
    # tokenizer = Tokenizer()
    # data = open('S:\\AI\\Musical_data_reviewtextonly_file.csv').read()
    # corpus = data.lower().split("\n")[:300]
    # tokenizer.fit_on_texts(corpus)
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
