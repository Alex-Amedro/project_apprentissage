import PIL
from PIL import Image
import csv
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam



# Chargement d'un dataset TensorFlow à partir du dossier d'images
train_dataset = tf.keras.utils.image_dataset_from_directory(
    'train/images',
    labels=None, # Mettre None si les labels sont dans un autre dossier
    image_size=(224, 224), # Redimensionner vos images si besoin
    batch_size=32
)