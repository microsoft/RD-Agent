import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)
print(tf.test.is_gpu_available())


def model_workflow(
    X: np.ndarray,
    y: np.ndarray,
    val_X: np.ndarray = None,
    val_y: np.ndarray = None,
    test_X: np.ndarray = None,
    **hyper_params,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """
    Manages the workflow of a machine learning model, including training, validation, and testing.

    If hyper_params is given, please get important hyperparameters from it. Otherwise, use the default values.
    (the hyper_params only contains important hyperparameters that is worth tunning)

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
    **hyper_params
        Additional hyperparameters for the model.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        Predictions on the validation data, predictions on the test data
    """
    train_images, train_labels = X, y
    validation_images, validation_labels = val_X, val_y
    test_images = test_X

    # Data augmentation is crucial for generalization, especially with small datasets.
    batch_size = hyper_params.get("batch_size", 64)

    train_datagen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True, vertical_flip=True)
    train_generator = train_datagen.flow(train_images, train_labels, batch_size=batch_size, shuffle=True)

    # Get input shape from the training data
    input_shape = X.shape[1:]
    num_classes = hyper_params.get("num_classes", 2)

    # Model Creation: Convolutional Neural Network
    dropout_dense_layer = hyper_params.get("dropout_dense_layer", 0.6)

    model = Sequential(
        [
            Conv2D(32, (3, 3), input_shape=input_shape),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(32, (3, 3)),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(32, (3, 3)),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3)),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(64, (3, 3)),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(64, (3, 3)),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3)),
            BatchNormalization(),
            Activation("relu"),
            Flatten(),
            Dense(1024),
            Activation("relu"),
            Dropout(dropout_dense_layer),
            Dense(256),
            Activation("relu"),
            Dropout(dropout_dense_layer),
            Dense(1),
            Activation("sigmoid"),
        ]
    )

    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(learning_rate=hyper_params.get("learning_rate", 0.001)),
        metrics=["accuracy"],
    )

    # Extract early_stop_round from hyper_params, default is 25
    early_stop_round = hyper_params.get("early_stop_round", 25)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=early_stop_round),
        ModelCheckpoint(filepath="best_model.keras", monitor="val_loss", save_best_only=True),
    ]

    # Training
    epochs = hyper_params.get("epochs", 100)
    if val_X is not None and val_y is not None:
        validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
        validation_generator = validation_datagen.flow(validation_images, validation_labels, batch_size=batch_size)
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            callbacks=callbacks,
        )
        # Dynamic adjustment of early_stop_round
        if "early_stop_round" not in hyper_params:
            val_loss = history.history["val_loss"]
            best_epoch = np.argmin(val_loss)
            dynamic_early_stop = max(5, int((len(val_loss) - best_epoch) * 0.5))  # 50% of remaining epochs

            print(f"Dynamic early_stop_round: {dynamic_early_stop}")
            hyper_params["early_stop_round"] = dynamic_early_stop

        # Predict on validation data
        val_pred = model.predict(validation_datagen.flow(validation_images, batch_size=1, shuffle=False), verbose=1)
    else:
        history = model.fit(
            train_generator,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            callbacks=callbacks,
        )
        val_pred = None

    # Predict on test data
    if test_X is not None:
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        test_generator = test_datagen.flow(test_images, batch_size=1, shuffle=False)
        test_pred = model.predict(test_generator, verbose=1)
    else:
        test_pred = None

    return val_pred, test_pred, hyper_params
