'''
As we saw before, categorical cross-entropy requires that we first integer-encode our categorical labels and then convert them 
to one-hot encodings using to_categorical(). There is another type of loss – sparse categorical cross-entropy – which is a 
computationally modified categorical cross-entropy loss that allows you to leave the integer labels as they are and skip the 
entire procedure of encoding.

Sparse categorical cross-entropy is mathematically identical to categorical cross-entropy but introduces some computational 
shortcuts that save time in memory as well as computation because it uses a single integer for a class, rather than a whole 
vector. This is especially useful when we have data with many classes to predict.
'''

import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  InputLayer
from tensorflow.keras.layers import  Dense
from sklearn.metrics import classification_report
import numpy as np
#your code here

train_data = pd.read_csv("air_quality_train.csv")
test_data = pd.read_csv("air_quality_test.csv")

#print columns and their respective types
print(train_data.info())
#print the class distribution
print(Counter(train_data["Air_Quality"]))
#extract the features from the training data
x_train = train_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the training data
y_train = train_data["Air_Quality"]
#extract the features from the test data
x_test = test_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the test data
y_test = test_data["Air_Quality"]

#encode the labels into integers
le = LabelEncoder()
#convert the integer encoded labels into binary vectors
y_train=le.fit_transform(y_train.astype(str))
y_test=le.transform(y_test.astype(str))
#convert the integer encoded labels into binary vectors
#we comment it here because we need only integer labels for

#these lines commented out because we are going to use sparse cross-entropy instead
#y_train = tensorflow.keras.utils.to_categorical(y_train, dtype = 'int64')
#y_test = tensorflow.keras.utils.to_categorical(y_test, dtype = 'int64')

#design the model
model = Sequential()
#add the input layer
model.add(InputLayer(input_shape=(x_train.shape[1],)))
#add a hidden layer
model.add(Dense(10, activation='relu'))
#add an output layer
model.add(Dense(6, activation='softmax'))

#compile the model, using sparse method
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # standard cross entropy

#train and evaluate the model
model.fit(x_train, y_train, epochs = 30, batch_size = 16, verbose = 0)

#get additional statistics
y_estimate = model.predict(x_test, verbose=0)
y_estimate = np.argmax(y_estimate, axis=1)
print(classification_report(y_test, y_estimate))


