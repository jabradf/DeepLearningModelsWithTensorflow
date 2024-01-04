import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy
import pandas as pd
import seaborn as sns

training_data_generator = ImageDataGenerator(zoom_range = 0.2,
                                             rotation_range = 15,
                                             width_shift_range = 0.05, 
                                             height_shift_range = 0.05,
                                             rescale=1.0/255)

Directory = 'project_x_ray_lungs/Covid19-dataset/train'

training_iterator = training_data_generator.flow_from_directory(
    Directory,
    class_mode = 'categorical',
    color_mode = 'grayscale',
    target_size = (256,256),
    batch_size = 32)
    
valid_data_generator = ImageDataGenerator(rescale=1.0/255)

Dir_valid = 'project_x_ray_lungs/Covid19-dataset/test'

valid_iterator = valid_data_generator.flow_from_directory(
    Dir_valid,
    class_mode = 'categorical',
    color_mode = 'grayscale',
    target_size = (256,256),
    batch_size = 12)
    
####### MODEL #######

model = Sequential()

model.add(tf.keras.Input(shape=(256,256,1)))

model.add(tf.keras.layers.Conv2D(3,5,strides=3,activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=(3,3)))

model.add(tf.keras.layers.Conv2D(4,3,strides=1,activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(4,4),strides=(4,4)))

model.add(tf.keras.layers.Conv2D(3,3,strides=1,activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(3,activation='softmax'))


print(model.summary())

#early_Stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.008),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy(),
                       tf.keras.metrics.AUC()])#,
              #callbacks=[early_Stop])

history = model.fit(
    training_iterator,
    steps_per_epoch = len(training_iterator)/12,
    epochs = 60,
    validation_data = valid_iterator,
    validation_steps = len(valid_iterator)/8
    
)