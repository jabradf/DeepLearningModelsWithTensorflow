import tensorflow as tf

print("\n\nModel with 8 filters:")

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(256, 256, 1)))

#Adds a Conv2D layer with 8 filters, each size 3x3:
model.add(tf.keras.layers.Conv2D(16, 7,activation="relu"))
model.summary()

#########

print("Model with 16 filters:")

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(256, 256, 1)))

#Adds a Conv2D layer with 16 filters, each size 7x7, and uses a stride of 1 with valid padding:
# stride is how it shifts to the next pixel (ie 1 at a time). Decreases the output dimensions with increasing value
# padding is defining what happens at the end of a row/column (stop/valid or add zeros/same pixel info: same)
model.add(tf.keras.layers.Conv2D(16, 7,
strides=1,
padding="same",
activation="relu"))
model.summary()