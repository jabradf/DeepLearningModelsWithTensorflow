import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
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
x_train = train_data.iloc[:,0:-1]
#x_train = train_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the training data
y_train = train_data["Air_Quality"]
#extract the features from the test data
x_test = test_data.iloc[:,0:-1]
#x_test = test_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the test data
y_test = test_data["Air_Quality"]

#encode the labels into integers
le = LabelEncoder()
#convert the integer encoded labels into binary vectors
y_train=le.fit_transform(y_train.astype(str))
y_test=le.transform(y_test.astype(str))
#convert the integer encoded labels into binary vectors
y_train = tensorflow.keras.utils.to_categorical(y_train, dtype = 'int64')
y_test = tensorflow.keras.utils.to_categorical(y_test, dtype = 'int64')

#design the model
model = Sequential()

#add the input layer
model.add(InputLayer(input_shape=(x_train.shape[1],)))

#add a hidden layer
'''
This layer has ten hidden units and uses a rectified linear unit (relu) as the activation function.
'''
model.add(Dense(10, activation='relu'))

#add an output layer
'''
Finally, we need to set the output layer. For regression, we don’t use any activation function in the final layer because we needed to predict a number without any transformations. However, for classification, the desired output is a vector of categorical probabilities.
'''
'''
To have this vector as an output, we need to use the softmax activation function that outputs a vector with elements having values between 0 and 1 and that sum to 1 (just as all the probabilities of all outcomes for a random variable must sum up to 1). In the case of a binary classification problem, a sigmoid activation function can also be used in the output layer but paired with the binary_crossentropy loss.

Since we have 6 classes to predict in our glass production data, the final softmax layer must have 6 units:
'''
model.add(Dense(6, activation='softmax')) #the output layer is a softmax with 3 units

#compile the model with cross entropy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train and evaluate the model
model.fit(x_train, y_train, epochs=20, batch_size=4, verbose=1)
#loss, acc = model.evaluate(x_test, y_test, verbose=0)

#get additional statistics
y_estimate = model.predict(x_test)
y_estimate = np.argmax(y_estimate, axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_estimate))