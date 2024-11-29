import numpy as np
from keras.utils import to_categorical
from tensorflow.keras.preprocessing import text, sequence
import pandas as pd

def prepare_data(data):
    x, y = data['text'].values, data['author'].values
    return x, y

def encode_labels(y_train, y_valid):
    label_dict = {'EAP': 0, 'HPL': 1, 'MWS': 2}
    y_train = pd.Series(y_train).replace(label_dict, inplace=False).values
    y_valid = pd.Series(y_valid).replace(label_dict, inplace=False).values
    return y_train, y_valid

def binarize_labels(y_train, y_valid):
    y_train_matrix = to_categorical(y_train)
    y_valid_matrix = to_categorical(y_valid)
    return y_train_matrix, y_valid_matrix

def tokenize_text(x_train, x_valid):
    token = text.Tokenizer(num_words=None)
    token.fit_on_texts(list(x_train) + list(x_valid))
    word_index = token.word_index
    x_train_seq = token.texts_to_sequences(x_train)
    x_valid_seq = token.texts_to_sequences(x_valid)
    return x_train_seq, x_valid_seq, word_index

def pad_sequences(x_train_seq, x_valid_seq, maxlen_):
    x_train_pad = sequence.pad_sequences(x_train_seq, maxlen=maxlen_, padding='pre', truncating='pre', value=0.0)
    x_valid_pad = sequence.pad_sequences(x_valid_seq, maxlen=maxlen_, padding='pre', truncating='pre', value=0.0)
    return x_train_pad, x_valid_pad

def load_glove_vectors(glove_path):
    embed_glove = {}
    with open(glove_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embed_glove[word] = vector
    return embed_glove

def create_embedding_matrix(word_index, embed_glove, dim_glove):
    word_vectorization_matrix = np.zeros((len(word_index) + 1, dim_glove))
    for word, i in word_index.items():
        word_embed_vector = embed_glove.get(word)
        if word_embed_vector is not None:
            word_vectorization_matrix[i] = word_embed_vector
    return word_vectorization_matrix