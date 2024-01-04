import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from utils import load_galaxy_data

#import app


input_data, labels = load_galaxy_data()

# Step 1
print("Input Data:")
print(input_data.shape)
# 1400 images, 128x128 in size, 3 channels of data (ie RGB)
print("Labels:")
print(labels.shape)
# 1400 images, 4 categories

# Step 2
x_train, x_valid, y_train, y_valid = train_test_split(input_data, labels, test_size=0.20, stratify=labels, shuffle=True, random_state=222)

# Step 3: preprocess the input
data_generator = ImageDataGenerator(rescale=1./255)

# Step 4: numpy iterators
batchSize = 5
training_iterator = data_generator.flow(x_train, y_train,batch_size=batchSize)
validation_iterator = data_generator.flow(x_valid, y_valid, batch_size=batchSize)

# Step 5&7: Create the model
model = tf.keras.Sequential() # s5
model.add(tf.keras.Input(shape=(128, 128, 3)))  #s5
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=(2,2)))
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16,activation="softmax")) #s7
model.add(tf.keras.layers.Dense(4,activation="softmax"))  #s5, output 4 features


# Step 6: Compile the model
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()]
)

# Step 8
model.summary()

# Step 9: Train the model
# we don't use the '.samples' method as we're using numpy here, so take the len()
model.fit(
        training_iterator,
        steps_per_epoch = len(training_iterator)/batchSize,
        epochs=15,
        validation_data = validation_iterator,
        validation_steps = len(validation_iterator)/batchSize)

# The model achieves ok accuracy at 61%, and AUC of 0.83.Not too bad for a classification system that predicts 4 types of images.

'''
Things that we could adjust to try improve the model:
* learning rate
* number of convolutional layers
* number of filters, strides, and padding type per layer
* stride and pool_size of max pooling layers
* size of hidden linear layers
'''

# Step 12: bonus visualisation
from visualise import visualise_activations
visualise_activations(model,validation_iterator)