'''
Whenever a convolutional layer is used, a Flatten layer needs to be used between the convolutional and the dense layers.

This is because dense layers apply their matrix to the dimension.
'''

import tensorflow as tf

model = tf.keras.Sequential()


model.add(tf.keras.Input(shape=(256,256,1)))

#Add a Conv2D layer
# - with 2 filters of size 5x5
# - strides of 3
# - valid padding
model.add(tf.keras.layers.Conv2D(2, 5, strides=3, padding="valid", activation="relu"))

# add a stacked layer
model.add(tf.keras.layers.Conv2D(4, 3, strides=1, padding="valid", activation="relu"))

model.add(tf.keras.layers.Flatten())

# #Remove these two dense layers:
#model.add(tf.keras.layers.Dense(100,activation="relu"))
#model.add(tf.keras.layers.Dense(50,activation="relu"))

model.add(tf.keras.layers.Dense(2,activation="softmax"))

#Print model information:
model.summary()