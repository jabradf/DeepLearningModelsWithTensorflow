BUILD DEEP LEARNING MODELS WITH TENSORFLOW
# Deep Learning Regression with Admissions Data
## Project Goals
For this project, you will create a deep learning regression model that predicts the likelihood that a student applying to graduate school will be accepted based on various application factors (such as test scores).

By analyzing the parameters in this [graduate admissions dataset](https://www.kaggle.com/mohansacharya/graduate-admissions?select=Admission_Predict_Ver1.1.csv), you will use TensorFlow with Keras to create a regression model that can evaluate the chances of an applicant being admitted. You hope this will give you further insight into the graduate admissions world and improve your test prep strategy.

## Project Requirements
1. If you take a look at **admissions_data.csv**, you’ll see parameters that admissions officers commonly use to evaluate university applicants. This data is from [Kaggle](https://www.kaggle.com/mohansacharya/graduate-admissions?select=Admission_Predict_Ver1.1.csv) and provides information about 500 applications for various universities and what their chance of admittance is.

This is a regression problem because the probability of being admitted is a continuous label between 0 and 1.

Load the **csv** file into a DataFrame and investigate the rows and columns to get familiarity with the dataset.

To get more information about each parameter in **admissions_data.csv**.


2. Split it up the data into `feature` parameters and the `labels`.

You are creating a model that predicts an applicant’s likelihood of being admitted to a master’s program, so take some time to look at the features of your model and which column you are trying to predict. Also consider if there are any dataset features that should not be included as a predictor.

Make sure all of your variables are numerical.

If there are any categorical variables, be sure to map them to numerical values, using techniques such as one-hot-encoding, so they can be used in a regression analysis.


3. Since you are creating a learning model, you must have a training set and a test set. Remember that this allows you to measure the effectiveness of your model.

You have created two DataFrames: one for `features` DataFrame and one for `labels`. Now, you must split each of these into a training set and a test set.

If you need a refresher on splitting a train and test set, use [scikit-learn’s user guide](https://scikit-learn.org/stable/user_guide.html) and any other online resources for help!

Click the hint below if you need any other guidance.


4. If you look through the **admissions_data.csv**, you may notice that there are many different scales being used. For example, the `GRE Score` is out of 340 while the `University Rating` is out of 5. Can you imagine why this might be a problem when using a regression learning model?

You should either scale or normalize your data so that all columns/features have equal weight in the learning model.


5. Create a neural network model to perform a regression analysis on the admission data.

When designing your own neural network model, consider the following:

* The shape of your input
* Adding hidden layers as well as how many neurons they have
* Including activation functions
* The type of loss function and metrics you use
* The type of gradient descent optimizer you use
* Your learning rate

6. It’s time to test out the model you created!

Fit your model with your training set and test it out with your test set.

It’s okay if it is not that accurate right now. You can play around with your model and tweak it to increase its accuracy.


7. You have tested out your model. Now is the time to adjust your model’s hyperparameters. You have a lot of choices to make. You can choose:

* the number of `epochs`
* the size of your `batch_size`
* to add more hidden layers
* your type of optimizer and/or activation functions.

Have fun in the hyperparameter playground. Test things out and see what works and what does not work. See what makes your model optimized between speed and accuracy. You have complete creative power!


## Solution
8. Great work! Visit our [forums](https://discuss.codecademy.com/c/project) to share your project with other learners. We recommend hosting your own solution on GitHub so you can share it more easily.

Your solution might look different from ours, and that’s okay. There are multiple ways to solve these projects, and you’ll learn more by seeing others’ code.

## Extensions
9. Using the [Matplotlib](https://matplotlib.org/) Library , see if you can plot the model loss per epoch as well as the mean-average error per epoch for both training and validation data. This will give you an insight into how the model performs better over time and can also help you figure out better ways to tune your hyperparameters.

Because of the way Matplotlib plots are displayed in the learning environment, please use `fig.savefig('static/images/my_plots.png')` at the end of your graphing code to render the plot in the browser. If you wish to display multiple plots, you can use `.subplot()` or `.add_subplot()` methods in the Matplotlib library to depict multiple plots in one figure.

Use the hint below if you have any struggles with displaying these graphs.


10. Let’s say you wanted to evaluate how strongly the features in **admissions.csv** predict an applicant’s admission into a graduate program. We can use something called an R-squared value. It is also known as the coefficient of determination; feel free to explore more about it [here](https://en.wikipedia.org/wiki/Coefficient_of_determination).

Basically, we can use this calculation to see how well the features in our regression model make predictions. An R-squared value near close to 1 suggests a well-fit regression model, while a value closer to 0 suggests that the regression model does not fit the data well.

See if you can apply this to your model after it has been evaluated using a `.predict()` method on your `features_test_set` and the `r2_score()` function on your `labels_test_set`. Both of these functions are from the [scikit-learn library](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html).