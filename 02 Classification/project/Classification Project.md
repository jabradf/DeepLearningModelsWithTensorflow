BUILD DEEP LEARNING MODELS WITH TENSORFLOW
# Classification
In this project, you will use [a dataset from Kaggle](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) to predict the survival of patients with heart failure from serum creatinine and ejection fraction, and other factors such as age, anemia, diabetes, and so on.

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Heart failure is a common event caused by CVDs, and this dataset contains 12 features that can be used to predict mortality by heart failure.

Most cardiovascular diseases can be prevented by addressing behavioral risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity, and harmful alcohol use using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidemia, or already established disease) need early detection and management wherein a machine learning model can be of great help.

## Loading the data
1. Using `pandas.read_csv()`, load the data from **heart_failure.csv** to a pandas DataFrame object. Assign the resulting DataFrame to a variable called `data`.


2. Use the `DataFrame.info()` method to print all the columns and their types of the DataFrame instance `data`.


3. Print the distribution of the `death_event` column in the `data` DataFrame class using `collections.Counter`. This is the column you will need to predict.



4. Extract the label column death_event from the `data` `DataFrame` and assign the result to a variable called `y`.



5. Extract the features columns `['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']` from the DataFrame instance `data` and assign the result to a variable called `x`.


## Data preprocessing
6. Use the `pandas.get_dummies()` function to convert the categorical features in the DataFrame instance `x` to one-hot encoding vectors and assign the result back to variable `x`.



7. Use the `sklearn.model_selection.train_test_split()` method to split the data into training features, test features, training labels, and test labels, respectively. To the `test_size` parameter assign the percentage of data you wish to put in the test data, and use any value for the `random_state` parameter. Store the results of the function to `X_train`, `X_test`, `Y_train`, `Y_test` variables, making sure you use this order.



8. Initialize a ColumnTransformer object by using `StandardScaler` to scale the numeric features in the dataset: `['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']`. Assign the resulting object to a variable called `ct`.



9. Use the `ColumnTransformer.fit_transform()` function to train the scaler instance ct on the training data `X_train` and assign the result back to `X_train`.



10. Use the `ColumnTransformer.transform()` to scale the test data instance `X_test` using the trained scaler `ct`, and assign the result back to `X_test`.



## Prepare labels for classification
11. Initialize an instance of `LabelEncoder` and assign it to a variable called `le`.



12. Using the `LabelEncoder.fit_transform()` function, fit the encoder instance `le` to the training labels `Y_train`, while at the same time converting the training labels according to the trained encoder.



13. Using the `LabelEncoder.transform()` function, encode the test labels `Y_test` using the trained encoder `le`.



14. Using the `tensorflow.keras.utils.to_categorical()` function, transform the encoded training labels `Y_train` into a binary vector and assign the result back to `Y_train`.



15. Using the `tensorflow.keras.utils.to_categorical()` function, transform the encoded test labels `Y_test` into a binary vector and assign the result back to `Y_test`.



## Design the model
16. Initialize a `tensorflow.keras.models.Sequential` model instance called `model`.



17. Create an input layer instance of `tensorflow.keras.layers.InputLayer` and add it to the model instance model using the `Model.add()` function.



18. Create a hidden layer instance of `tensorflow.keras.layers.Dense` with `relu` activation function and 12 hidden neurons, and add it to the model instance `model`.



19. Create an output layer instance of `tensorflow.keras.layers.Dense` with a `softmax` activation function (because of classification) with the number of neurons corresponding to the number of classes in the dataset.


20. Using the `Model.compile()` function, compile the model instance model using the `categorical_crossentropy` loss, `adam` optimizer and `accuracy` as metrics.


## Train and evaluate the model
21. Using the `Model.fit()` function, fit the model instance model to the training data `X_train` and training labels `Y_train`. Set the number of epochs to 100 and the batch size parameter to 16.


22. Using the `Model.evaluate()` function, evaluate the trained `model` instance model on the test data `X_test` and test labels `Y_test`. Assign the result to a variable called `loss` (representing the final loss value) and a variable called `acc` (representing the accuracy metrics), respectively.


## Generating a classification report
23. Use the `Model.predict()` to get the predictions for the test data `X_test` with the trained model instance model. Assign the result to a variable called `y_estimate`.


24. Use the `numpy.argmax()` method to select the indices of the true classes for each label encoding in `y_estimate`. Assign the result to a variable called `y_estimate`.

25. Use the `numpy.argmax()` method to select the indices of the true classes for each label encoding in `Y_test`. Assign the result to a variable called `y_true`.

26. Print additional metrics, such as F1-score, using the `sklearn.metrics.classification_report()` function by providing it with `y_true` and `y_estimate` vectors as input parameters.