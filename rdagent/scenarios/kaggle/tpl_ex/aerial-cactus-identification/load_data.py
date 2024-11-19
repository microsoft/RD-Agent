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


def load_from_raw_data():
    """
    load raw data from disk to get data in uniform data

    Return:
        train_images: np.array
        train_labels: np.array
        test_images: np.array
        test_ids: the id representing the image. it is used to generate the submission file
    """
    images, labels = load_images_and_labels("/kaggle/input/train.csv", "/kaggle/input/train/")

    test_folder = "/kaggle/input/test/"
    test_images, test_filenames = load_test_images(test_folder)
    # Store filenames separately
    test_filenames = [os.path.basename(filename).replace(".tif", "") for filename in test_filenames]

    training_df = pd.read_csv("/kaggle/input/train.csv")
    training_df.head()

    return images, labels, test_images, test_filenames
