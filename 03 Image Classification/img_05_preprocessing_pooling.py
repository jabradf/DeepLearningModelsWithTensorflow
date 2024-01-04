'''
Pooling helps reduce the size of hidden layers (and reducing overfitting). 
These layers have another useful property: they provide some amount of translational invariance. 
In other words, even if we move around objects in the input image, the output will be the same. 

This is very useful for classification. For example, we usually want to classify an image of a cat as a cat, 
regardless of how the cat is oriented in the image.
'''

import tensorflow as tf

model = tf.keras.Sequential()


model.add(tf.keras.Input(shape=(256,256,1)))

model.add(tf.keras.layers.Conv2D(2,5,strides=3,padding="valid",activation="relu"))

#Add first max pooling layer here.
model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 5),   strides=(5, 5), padding='valid'))


model.add(tf.keras.layers.Conv2D(4,3,strides=1,padding="valid",activation="relu"))

#Add the second max pooling layer here.
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),   strides=(2, 2), padding='valid'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(2,activation="softmax"))

#Print model information:
model.summary() 

'''
Before adding the pooling layers, the numer of parameters was 53,922.
Afterwards, the number drops to 522! 
We can see that the first max-pooling layer shrinks height and width of its input shape by a factor of 4. 
The second max-pooling layer shrinks its input shape by a factor of 2.
'''