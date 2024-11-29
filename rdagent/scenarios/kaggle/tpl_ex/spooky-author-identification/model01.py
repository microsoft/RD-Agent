import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding, SpatialDropout1D
from keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

def multiclass_log_loss(y_true_binarized, y_pred_probabilities, epsilon=1e-20):
    y_pred_probabilities = K.clip(y_pred_probabilities, epsilon, 1 - epsilon)
    sum_ = tf.cast(K.sum(y_true_binarized * K.log(y_pred_probabilities)), tf.float64)
    logloss = (-1 / len(y_true_binarized)) * sum_
    return logloss

def build_model(word_index, dim_glove, word_vectorization_matrix, maxlen_):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, dim_glove, weights=[word_vectorization_matrix], input_length=maxlen_, trainable=False))
    model.add(SpatialDropout1D(0.3))
    model.add(LSTM(dim_glove, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(3, activation='softmax'))
    return model

def compile_model(model, initial_learning_rate):
    model.compile(loss=multiclass_log_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate))

def train_model(model, x_train_pad, y_train_matrix, x_valid_pad, y_valid_matrix, initial_learning_rate):
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=1, mode='auto', start_from_epoch=60)
    
    def scheduler_modified_exponential(epoch, learning_rate):
        if epoch < 40:
            return learning_rate
        else:
            return learning_rate * math.exp(-0.1)
    
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler_modified_exponential)
    
    history = model.fit(x_train_pad, y=y_train_matrix, batch_size=256, epochs=100, verbose=0, validation_data=(x_valid_pad, y_valid_matrix), callbacks=[earlystop, learning_rate_scheduler])
    return history

def evaluate_model(model, x_train_pad, y_train_matrix, x_valid_pad, y_valid_matrix):
    y_train_matrix_pred = model.predict(x_train_pad)
    y_valid_matrix_pred = model.predict(x_valid_pad)
    logloss_train = multiclass_log_loss(y_train_matrix, y_train_matrix_pred).numpy()
    logloss_valid = multiclass_log_loss(y_valid_matrix, y_valid_matrix_pred).numpy()
    return logloss_train, logloss_valid, y_train_matrix_pred, y_valid_matrix_pred

def calculate_accuracy(y_true, y_pred):
    match = (y_true == y_pred)
    match = np.array(list(map(lambda x: int(x == True), match)))
    accuracy = round(match.sum() / len(match), 3)
    return accuracy