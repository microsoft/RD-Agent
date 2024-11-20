import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Reshape, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



print(tf.__version__)
print(tf.test.is_gpu_available())

def model_workflow(X: np.ndarray,
                   y: np.ndarray,
                   val_X: np.ndarray = None,
                   val_y: np.ndarray = None,
                   test_X: np.ndarray = None,
                   **hyper_param) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """
    A function to manage the workflow of a machine learning model, including training, validation, and testing.

    Parameters
    ----------
    X : np.ndarray
        Training data features.
        
    y : np.ndarray
        Training data labels.
        
    val_X : np.ndarray, optional
        Validation data features.
        
    val_y : np.ndarray, optional
        Validation data labels.
        
    test_X : np.ndarray, optional
        Test data features.
        
    test_y : np.ndarray, optional
        Test data labels.
        
    **hyper_param
        Additional hyperparameters for the model.

    Returns
    -------
    np.ndarray
        Predictions on the val data.

    np.ndarray
        Predictions on the test data.

    dict
        A dictionary containing updated hyperparameters after model training and evaluation.
    """
    train_images, train_labels, validation_images, validation_labels = X, y, val_X, val_y
    test_images = test_X

    # The dataset being relatively small, data augmentation is very important to generalise and learn what a cactus look like. Based on the fact that cactus detection seems like an easy problem and we're dealing with a small amount of data, the batch size is kept small as training will be quick anyway.

    # In[7]:

    batch_size = 64

    train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, vertical_flip=True)

    train_generator = train_datagen.flow(train_images, train_labels, batch_size=batch_size, shuffle=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow(validation_images, validation_labels, batch_size=batch_size)

    input_shape = (32, 32, 3)
    num_classes = 2

    # # Model Creation: Convolutional Neural Network

    # Some really insightful comments about deep learning model optimization can be found here (https://karpathy.github.io/2019/04/25/recipe/ ).

    # In[8]:

    dropout_dense_layer = 0.6

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_dense_layer))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_dense_layer))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # In[9]:

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    # In[10]:

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25),
        ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True)
    ]

    # # Training

    # In[11]:

    epochs = 2
    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        epochs=epochs,
                        verbose=1,
                        shuffle=True,
                        callbacks=callbacks)

    # # Load the test data and evaluate the model

    # Load the best performing model based on the validation loss.

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow(test_images, batch_size=1, shuffle=False)


    test_pred = model.predict(test_generator, verbose=1)
    return None, test_pred, hyper_param
