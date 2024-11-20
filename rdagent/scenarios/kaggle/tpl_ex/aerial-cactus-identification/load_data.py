"""
Load competition data to uniform format
"""

import os

import numpy as np
import pandas as pd
from PIL import Image


def load_test_images(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(np.array(img))
            filenames.append(filename)
    return np.array(images), filenames


def load_images_and_labels(csv_file, image_folder):
    images = []
    labels = []
    df = pd.read_csv(csv_file)
    for idx, row in df.iterrows():
        img = Image.open(os.path.join(image_folder, row["id"]))
        if img is not None:
            images.append(np.array(img))
            labels.append(row["has_cactus"])
    return np.array(images), np.array(labels)


def load_from_raw_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    load raw data from disk to get data in uniform data

    Return:
        X: np.array

            a concrete example could be:

            .. code-block:: text

                array([[[[207, 194, 203],
                        ...,
                        [191, 183, 164],
                        [176, 168, 149],
                        [181, 173, 152]]]], dtype=uint8)

        y: np.array

            a concrete example could be:

            .. code-block:: python

                array([1, 0, 1, 0, 1, 1, ..., ])

        X_test: np.array

            a concrete example is similar to `X`.

        test_ids: the id representing the image. it is used to generate the submission file

            a concrete example could be:

            .. code-block:: python

                ['1398ad045aa57aee5f38e7661e9d49e8.jpg',
                '0051207eb794887c619341090de84b50.jpg',
                'a8202dd82c42e252bef921ada7607b6c.jpg',
                '76c329ff9e3c5036b616f4e88ebba814.jpg',
                ...]
    """
    X, y = load_images_and_labels("/kaggle/input/train.csv", "/kaggle/input/train/")

    test_folder = "/kaggle/input/test/"
    X_test, test_filenames = load_test_images(test_folder)
    # Store filenames separately
    test_ids = [os.path.basename(filename).replace(".tif", "") for filename in test_filenames]
    return X, y, X_test, test_ids
