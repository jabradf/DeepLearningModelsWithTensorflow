#import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score

def design_model(features, numNeurons=32):
  #not used
  my_model = Sequential(name = "my_first_model")
  input = InputLayer(input_shape=(features.shape[1],)) 
  my_model.add(input)

  #add hidden layer
  my_model.add(Dense(numNeurons, activation='relu')) 
  my_model.add(Dense(1)) 
  # optimiser layer
  opt = Adam(learning_rate=0.01)
  my_model.compile(loss='mse',  metrics=['mae'], optimizer=opt)
  return my_model

dataset = pd.read_csv('admissions_data.csv') 
#print(dataset.columns)

# Step 2: Split into labels and features, 
# serial no has no bearing on the prediction, so we won't use it
labels = dataset.iloc[:,-1] #choose the final column for prediction
features = dataset.iloc[:,1:-1] #choose every other column as features

# all columns are float or int, no categoricals so no one-hot encoding needed
# Step 3: split into test and training sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=39) 

# Step 4: Normalise the data
from sklearn.compose import ColumnTransformer
columnSelect = features.select_dtypes(include=['float64', 'int64'])
columnSelectNum = columnSelect.columns
ct = ColumnTransformer([('transform', StandardScaler(), columnSelectNum)], remainder='passthrough')
train_scaled = ct.fit_transform(features_train)
test_scaled = ct.fit_transform(features_test)

# Step 5: create the model
numNeurons = 32
#model = design_model(features_train, numNeurons=64)
model = Sequential(name = "regression_model")
input = InputLayer(input_shape=(features.shape[1],)) 
model.add(input)

#add hidden layer
model.add(Dense(numNeurons, activation='relu')) 
model.add(Dense(1)) # output layer
print(model.summary())

# optimiser
opt = Adam(learning_rate=0.01)
model.compile(loss='mse',  metrics=['mae'], optimizer=opt)


# Step 6: Fit the data and train the model
stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)
data = model.fit(test_scaled, labels_test, epochs=200, batch_size=2, 
  verbose=1, validation_split = 0.15, callbacks=[stop])

res_mse, res_mae = model.evaluate(test_scaled, labels_test, verbose = 0)
print('Mean squared error (MSE) is the measure of learning success. The lower the loss the better the performance. It is the most common metric for regression problems.')
print("Loss, MSE: ", res_mse)
print('Mean Absolute Error (MAE) shows how far the predictions are from the true values on average.')
print("Metric, MAE: ", res_mae)

# Step 7 tune it!
# increased epochs, added early stopping with a little more patience (started at 20, now 40)

# Step 9 get R^2 score
y_pred = model.predict(test_scaled) #, verbose=0)
r_score = r2_score(labels_test, y_pred)
print("R2 score: ", r_score)
# R2 score 0 (or below) is bad fit, 1 is a good fit


# Do extensions code below
# if you decide to do the Matplotlib extension, you must save your plot in the directory by uncommenting the line of code below
fig = plt.figure(figsize=(14,10))
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(data.history['mae'])
ax1.plot(data.history['val_mae'])
ax1.set_title('Model MAE')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')
 
# Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(data.history['loss'])
ax2.plot(data.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')
 
# used to keep plots from overlapping each other  
fig.tight_layout()
plt.show()
#fig.savefig('static/images/my_plots.png')