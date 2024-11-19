#!/usr/bin/env python
# coding: utf-8

# ### The goal of this kernel is to create a simple Convolution Neural Network that will allow to differentiate images that contains cacti from images that do not. While better results closer to 100% accuracy could be designed using layers from pre-trained models such as VGG16, we want to stick with a fairly simple CNN architecture and see how far we can go, and whether pre-trained layers are even needed.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
from shutil import copyfile, move
from tqdm import tqdm
import h5py
from PIL import Image
from sklearn.model_selection import train_test_split

# In[2]:


# # Prepare the data

# In[3]:

from load_data import load_from_raw_data


train_images, train_labels, test_images, test_ids  = load_from_raw_data()

train_images, validation_images, train_labels, validation_labels = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=42)



from model01 import model_workflow
val_pred, test_pred, hyper_param = model_workflow(train_images, train_labels, validation_images, validation_labels)


from ens import ens_and_decision
pred_binary = ens_and_decision(test_pred, val_pred, validation_labels)
# In[17]:


csv_file = open("submission.csv","w")
csv_file.write("id,has_cactus\n")
for tid, prediction in zip(test_ids, pred_binary):
    csv_file.write(str(tid)+","+str(prediction)+"\n")
csv_file.close()

