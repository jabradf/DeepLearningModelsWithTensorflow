BUILD DEEP LEARNING MODELS WITH TENSORFLOW
# Implementing Neural Networks
The World Health Organization (WHO)’s Global Health Observatory (GHO) data repository tracks life expectancy for countries worldwide by following health status and many other related factors.

Although there have been a lot of studies undertaken in the past on factors affecting life expectancy considering demographic variables, income composition, and mortality rates, it was found that the effects of immunization and human development index were not taken into account.

This dataset covers a variety of indicators for all countries from 2000 to 2015 including:

* immunization factors
* mortality factors
* economic factors
* social factors
* other health-related factors

Ideally, this data will eventually inform countries concerning which factors to change in order to improve the life expectancy of their populations. If we can predict life expectancy well given all the factors, this is a good sign that there are some important patterns in the data. Life expectancy is expressed in years, and hence it is a number. This means that in order to build a predictive model one needs to use regression.

In this project, you will design, train, and evaluate a neural network model performing the task of regression to predict the life expectancy of countries using this dataset. Excited? Let’s go!

## Data loading and observing
1. Load the __life_expectancy.csv__ dataset into a pandas DataFrame by first importing pandas, and then using the `pandas.read_csv()` function to load the file and assign the resulting DataFrame to a variable called `dataset`.

2. Observe the data by printing the first five entries in the DataFrame `dataset` by using the `DataFrame.head()` function. Make sure to see what the columns are and what the data types are. Locate the feature we would like to predict: `life expectancy`. Use `DataFrame.describe()` to see the summary statistics of the data.

3. Drop the Country column from the DataFrame using the `DataFrame` drop method. Why? To create a predictive model, knowing from which country data comes can be confusing and it is not a column we can generalize over. We want to learn a general pattern for all the countries, and not only those dependent on specific countries.

4. In the next two steps, you will split the data into labels and features. Labels are contained in the “Life expectancy” column. It’s the final column in the DataFrame. Create a new variable called `labels`. Use `iloc` indexing to assign the final column of dataset to it.


5. Features span from the first column to the last column (not including it, since it’s a label column in our dataset). Create a new variable called `features`. Use `iloc` indexing to assign a subset of columns from first to last (not including the last column) to it.

## Data Preprocessing
6. When you observed your dataset you probably noticed that some columns are categorical. We learned in this lesson that categorical columns need to be converted into numerical columns using methods such as one-hot-encoding. Use `pandas.get_dummies(DataFrame)` to apply one-hot-encoding on all the categorical columns. Assign the result of the encoding back to the `features` variable.


7. Split your data into training set and test sets using the `sklearn.model_selection.train_test_split()` function. Assign

* the training features to a variable called `features_train`
* training labels to a variable called `labels_train`
* test data to a variable called `features_test`
* test labels to a variable called `labels_test`.

You can choose any percentage for `test_size` and any value for `random_state`.

8. The next step is to standardize/normalize your numerical features. You can pick whichever method you want. In this step, create a `sklearn.compose.ColumnTransformer` instance called `ct` to set up the normalization/standardization procedure. When initializing `ColumnTransformer` make sure to list all of the numerical features you have in your dataset. Or use `DataFrame.select_dtypes()` to select float64 or int64 feature types automatically.

9. Fit your instance `ct` of `ColumnTransformer` to the training data and at the same time transform it by using the `ColumnTransformer.fit_transform()` method. Assign the result to a variable called `features_train_scaled`.


10. Transform your test data instance `features_test` using the trained `ColumnTransformer` instance `ct`. Assign the result to a variable called `features_test_scaled`.


## Building the model
11. Create an instance of the `tensorflow.keras.models.Sequential` model called `my_model`.


12. Create the input layer to the network model using `tf.keras.layers.InputLayer` with the shape corresponding to the number of features in your dataset. Assign the resulting input layer to a variable called `input`.


13. Add the input layer from the previous step to the model instance `my_model`.


14. Add one `keras.layers.Dense` hidden layer with any number of hidden units you wish. Use the `relu` activation function.

15. Add a `keras.layers.Dense` output layer with one neuron since you need a single output for a regression prediction.

16. Print the summary of the model using the `Sequential.summary()` method.

## Initializing the optimizer and compiling the model
17. Create an instance of the `Adam` optimizer with the learning rate equal to 0.01.


18. Once your optimizer is initialized, compile the model using the `Sequential.compile()` method.

Assign the following parameters:

* For `loss` use the Mean Squared Error (`mse`)
* For `metrics` use the Mean Absolute Error (`mae`)
* For `opt` (the optimizer parameters) use the instance of the optimizer you created in the previous step.

## Fit and evaluate the model
19. Train your model with the `Sequential.fit()` method by passing it the following parameters:

* your preprocessed training data 
* training labels
* number of epochs equal to 40
* batch size equal to 1
* verbose set to 1

20. Using the `Sequential.evaluate()` method, evaluate your trained model on the preprocessed test data set, and with the test labels. Set verbose to 0. Store the result of the evaluation in two variables: `res_mse` and `res_mae`.


21. Print your final loss (RMSE) and final metric (MAE) to check the predictive performance on the test set.