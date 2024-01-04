'''
One way to classify image data is to treat an image as a vector of pixels. After all, we pass most data into our neural networks as feature vectors, so why not do the same here?

Two steps here:
1. Change the shape to accept the image data
2. flatten the layer to output it as a single vector
'''

import tensorflow as tf

model = tf.keras.Sequential()

#Add an input layer that will expect grayscale input images of size 256x256:
model.add(tf.keras.Input(shape=(256,256,1)))

#Use a Flatten() layer to flatten the image into a single vector:
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(100,activation="relu"))
model.add(tf.keras.layers.Dense(50,activation="relu"))
model.add(tf.keras.layers.Dense(2,activation="softmax"))

#Print model information:
model.summary() 

'''
Uncommenting the summary line shows that we have over 6.5M parameters, this is 1000x more than the data points and will result in overfitting, as well as significant compute time.

Next up is an alternative approach for extracting meaningful features; convolutional layers
'''

'''
Output is:

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 65536)             0         
_________________________________________________________________
dense (Dense)                (None, 100)               6553700   
_________________________________________________________________
dense_1 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 102       
=================================================================
Total params: 6,558,852
Trainable params: 6,558,852
Non-trainable params: 0
__________________________
'''