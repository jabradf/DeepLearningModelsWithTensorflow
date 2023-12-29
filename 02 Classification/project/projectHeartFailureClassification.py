import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np

# Step 1 - load
data = pd.read_csv("heart_failure.csv")

# Step 2 - take a look
print(data.info())
 
# Step 3 - look at the distribution
print(Counter(data["death_event"])) # 203 no, 96 yes

# Step 4 - extract the label data
y = data["death_event"]
#print(data['DEATH_EVENT'].head(10))

# Step 5 - Extract the training data, minus the 2 death_event columns
x = data.iloc[:,0:-2]

# Step 6 - one-hot encode the categoricals
x  = pd.get_dummies(x)

# Step 7 - split the data into test and train sets
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 30)

# Step 8 - Create a scaler using the provided columns
ct = ColumnTransformer([('numeric', StandardScaler(), ['age', 'creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])], remainder='passthrough')

# Step 9 - fit the data training data using ColumnTransformer
X_train = ct.fit_transform(X_train)

# Step 10 - Scale the test data using ColumnTransformer
X_test = ct.fit_transform(X_test)

# Step 11 - Initialise a LabelEncoder
le = LabelEncoder()

# Step 12 - fit the training data to binary vectors
Y_train = le.fit_transform(Y_train.astype(str))

# Step 13 - fit the test data to binary vectors
Y_test = le.fit_transform(Y_test.astype(str))

# Step 14 - transform the encoded training labels to binary vectors
Y_train = to_categorical(Y_train, dtype = 'int64')

# Step 15 - transform the encoded test labels to binary vectors
Y_test = to_categorical(Y_test, dtype = 'int64')

# Step 16 - initialise the model
model = Sequential()

# Step 17 - create the input layer
model.add(InputLayer(input_shape=(X_train.shape[1],)))

# Step 18 - Create a hidden layer with relu and 12 neurons
model.add(Dense(12, activation='relu'))

# Step 19 - Create the output layer. 2 Possible result classes
model.add(Dense(2, activation='softmax')) 

# Step 20 - Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 21 - Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1)

# Step 22 - Evaluate the model
loss, acc = model.evaluate(X_test, Y_test, verbose=1)
print("Loss:", loss, "Accuracy:", acc)

# Step 23 - get the predictions from the model
y_estimate = model.predict(X_test, verbose=1)

# Step 24 - select the indices of 'true' classes for the estimation
y_estimate = np.argmax(y_estimate, axis=1)

# Step 25 - select the indices of 'true' classes for the test
y_true = np.argmax(Y_test, axis=1)

# Step 16 - Print the model's metrics
print(classification_report(y_true, y_estimate))