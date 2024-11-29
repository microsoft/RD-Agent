import os
import time
import psutil
import numpy as np
from sklearn.model_selection import train_test_split
from feat01 import prepare_data, encode_labels, binarize_labels, tokenize_text, pad_sequences, load_glove_vectors, create_embedding_matrix
from model01 import build_model, compile_model, train_model, evaluate_model, calculate_accuracy
from load_data import load_data

def main():
    start = time.time()
    process = psutil.Process(os.getpid())
    
    train_path = '/kaggle/input/train.csv'
    test_path = '/kaggle/input/test.csv'
    glove_path = '/kaggle/input/glove.6B.100d.txt'
    
    train_data, test_data = load_data(train_path, test_path)
    print(f"Memory usage : {train_data.memory_usage().sum()/(1024*1024):.2f} MB")
    print(f"Dataset shape: {train_data.shape}")
    
    x, y = prepare_data(train_data)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, stratify=y, random_state=40, test_size=0.2, shuffle=True)

    y_train, y_valid = encode_labels(y_train, y_valid)
    
    embed_glove = load_glove_vectors(glove_path)
    dim_glove = len(embed_glove['the'])
    
    y_train_matrix, y_valid_matrix = binarize_labels(y_train, y_valid)
    
    token = text.Tokenizer(num_words=None)
    token.fit_on_texts(list(x_train) + list(x_valid))
    word_index = token.word_index
    
    x_train_seq, x_valid_seq, word_index = tokenize_text(x_train, x_valid)
    
    len_train = [len(x_train_seq[i]) for i in range(len(x_train_seq))]
    len_valid = [len(x_valid_seq[i]) for i in range(len(x_valid_seq))]
    len_ = np.array(len_train + len_valid)
    maxlen_ = math.floor(len_.mean() + 2*len_.std()) + 1
    
    x_train_pad, x_valid_pad = pad_sequences(x_train_seq, x_valid_seq, maxlen_)
    
    word_vectorization_matrix = create_embedding_matrix(word_index, embed_glove, dim_glove)
    
    model = build_model(word_index, dim_glove, word_vectorization_matrix, maxlen_)
    compile_model(model, initial_learning_rate=0.001)
    
    history = train_model(model, x_train_pad, y_train_matrix, x_valid_pad, y_valid_matrix, initial_learning_rate=0.001)
    
    logloss_train, logloss_valid, y_train_matrix_pred, y_valid_matrix_pred = evaluate_model(model, x_train_pad, y_train_matrix, x_valid_pad, y_valid_matrix)
    
    print(f"Training logloss  : {round(logloss_train, 3)}")
    print(f"Validation logloss: {round(logloss_valid, 3)}")
    
    y_train_true = np.array([np.argmax(x) for x in y_train_matrix])
    y_valid_true = np.array([np.argmax(x) for x in y_valid_matrix])
    y_train_pred = np.array([np.argmax(x) for x in y_train_matrix_pred])
    y_valid_pred = np.array([np.argmax(x) for x in y_valid_matrix_pred])
    
    train_accuracy = calculate_accuracy(y_train_true, y_train_pred)
    valid_accuracy = calculate_accuracy(y_valid_true, y_valid_pred)
    
    print(f"Training accuracy  : {train_accuracy}")
    print(f"Validation accuracy: {valid_accuracy}")
    
    end = time.time()
    print(f"Runtime: {end - start} seconds")
    print(f"Memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()