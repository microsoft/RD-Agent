#!/usr/bin/env python
# coding: utf-8

# ### The goal of this kernel is to create a simple Convolution Neural Network that will allow to differentiate images that contains cacti from images that do not. While better results closer to 100% accuracy could be designed using layers from pre-trained models such as VGG16, we want to stick with a fairly simple CNN architecture and see how far we can go, and whether pre-trained layers are even needed.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow import keras

import os
from shutil import copyfile, move
from tqdm import tqdm
import h5py
from PIL import Image


# Just a quick check to make verify Tensorflow version and whether the GPU is found.

# In[2]:


print(tf.__version__)
print(tf.test.is_gpu_available())


# # Prepare the data

# In order to prepare the dataset and use Keras' ImageDataGenerator, it was decided to extract the images to a new folder where images are sorted into 2 folders, one for images with cacti, and one for images without. It is, therefore, required to load the .csv file provided to retrieve the groundtruth and copy the files in the right folder.

# In[3]:


training_df = pd.read_csv("/kaggle/input/train.csv")
training_df.head()


# In[4]:


src = "/kaggle/input/train/"
dst = "../sorted_training/"

os.mkdir(dst)
os.mkdir(dst+"true")
os.mkdir(dst+"false")

with tqdm(total=len(list(training_df.iterrows()))) as pbar:
    for idx, row in training_df.iterrows():
        pbar.update(1)
        if row["has_cactus"] == 1:
            copyfile(src+row["id"], dst+"true/"+row["id"])
        else:
            copyfile(src+row["id"], dst+"false/"+row["id"])


# I then extract the validation from the folder where I stored the training set. Note that this time, the files are moved and not just copied.

# In[5]:


src = "../sorted_training/"
dst = "../sorted_validation/"

os.mkdir(dst)
os.mkdir(dst+"true")
os.mkdir(dst+"false")

validation_df = training_df.sample(n=int(len(training_df)/10))

with tqdm(total=len(list(validation_df.iterrows()))) as pbar:
    for idx, row in validation_df.iterrows():
        pbar.update(1)
        if row["has_cactus"] == 1:
            move(src+"true/"+row["id"], dst+"true/"+row["id"])
        else:
            move(src+"false/"+row["id"], dst+"false/"+row["id"])


# # Load the dataset

# Even though it is considered good practice to import all libraries at the beginning of a script, I tend to import the ones related to the model architecture right before, as it makes them visible and gives expectations to a reader as to what to find in the architecture.

# In[6]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Reshape, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# The dataset being relatively small, data augmentation is very important to generalise and learn what a cactus look like. Based on the fact that cactus detection seems like an easy problem and we're dealing with a small amount of data, the batch size is kept small as training will be quick anyway.

# In[7]:


batch_size = 64

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in ['true', 'false']:
        path = os.path.join(folder, label)
        for filename in os.listdir(path):
            img = Image.open(os.path.join(path, filename))
            if img is not None:
                images.append(np.array(img))
                labels.append(1 if label == 'true' else 0)
    return np.array(images), np.array(labels)

train_images, train_labels = load_images_from_folder("../sorted_training")
validation_images, validation_labels = load_images_from_folder("../sorted_validation")

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    vertical_flip=True)

train_generator = train_datagen.flow(
    train_images, train_labels,
    batch_size=batch_size,
    shuffle=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow(
    validation_images, validation_labels,
    batch_size=batch_size)

input_shape = (32,32,3)
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


callbacks = [EarlyStopping(monitor='val_loss', patience=25),
             ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True)]


# # Training

# In[11]:


epochs = 100
history = model.fit(train_generator,
          validation_data=validation_generator,
          epochs=epochs,
          verbose=1,
          shuffle=True,
          callbacks=callbacks)



# # Load the test data and evaluate the model

# Load the best performing model based on the validation loss.

# In[14]:
def load_test_images(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(np.array(img))
            filenames.append(filename)
    return np.array(images), filenames

test_folder = "/kaggle/input/test/"
test_images, test_filenames = load_test_images(test_folder)

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow(
    test_images,
    batch_size=1,
    shuffle=False
)

# Store filenames separately
test_filenames = [os.path.basename(filename) for filename in test_filenames]



# In[16]:


pred=model.predict(test_generator,verbose=1)
pred_binary = [0 if value<0.50 else 1 for value in pred]  


# Generate the submission .csv file.

# In[17]:


csv_file = open("submission.csv","w")
csv_file.write("id,has_cactus\n")
for filename, prediction in zip(test_filenames, pred_binary):
    name = filename.replace(".tif", "")
    csv_file.write(str(name)+","+str(prediction)+"\n")
csv_file.close()

