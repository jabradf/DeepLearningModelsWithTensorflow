import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1
#load the dataset
dataset = pd.read_csv('life_expectancy.csv') 

# Step 2
#print(dataset.head())
#print(dataset.describe())
# find categorical columns
#print(dataset.dtypes)
# only column is "Status"

# Step 3
#drop the country column to remove any generalisations made on a per country basis
dataset.drop(columns=['Country'], inplace=True)

# Step 4 & 5
#split the dataset into labels and features
labels = dataset.iloc[:,-1] #choose the final column for prediction
features = dataset.iloc[:,0:-1] #choose first 7 columns as features

# Step 6
# Apply one-hot-encoding on the "Status" categorical column. May as well do everything at once:
features = pd.get_dummies(features) #one-hot encoding for categorical variables

# Step 7: split the data into test and train sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42) 

#step 8 normalise the data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
# automatically select the numeric columns. dtypes listed these columns as float64 and in64 so that's what we will use here
columnSelect = features.select_dtypes(include=['float64', 'int64'])
columnSelectNum = columnSelect.columns
ct = ColumnTransformer([('only numeric', StandardScaler(), columnSelectNum)], remainder='passthrough')

# Step 9&10: fit the data
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)

# Step 11: create the model
from tensorflow.keras.models import Sequential
my_model = Sequential(name = "my_first_model")
#design_model(features_train)
#print(my_model.summary())

# Step 12: create input layer
from tensorflow.keras.layers import InputLayer
# coded so that it is dynamic, not hard coded to the df shape
input = InputLayer(input_shape=(features.shape[1],)) 

# Step 13: add input layer to the model
my_model.add(input) 

# Step 14: add hidden layer
from tensorflow.keras.layers import Dense
neurons = 64
my_model.add(Dense(neurons, activation='relu')) 

# Step 15 & 16: Add output layer using Dense, and print model summary
my_model.add(Dense(1)) 
print(my_model.summary())

# Step 17: optimise using Adam, learning rate 0.01
from tensorflow.keras.optimizers import Adam
opt = Adam(learning_rate=0.01)

# Step 18: compile the model
my_model.compile(loss='mse',  metrics=['mae'], optimizer=opt)

# Step 19: train the model
my_model.fit(features_train, labels_train, epochs=40, batch_size=1, verbose=1)

# Step 20: evaluate the model
res_mse, res_mae = my_model.evaluate(features_test, labels_test, verbose = 0)

print("Final loss: ", res_mse)
print("Final metric: ", res_mae)