BUILD DEEP LEARNING MODELS WITH TENSORFLOW
# Classify X-rays With Machine Learning
## Project Goals
You are a researcher in a hospital lab and are given the task to develop a learning model that supports doctors with diagnosing illnesses that affect patients’ lungs. At your disposal, you have a set [X-ray lung scans](https://www.kaggle.com/pranavraikokte/covid19-image-dataset) with examples of patients who had either pneumonia, Covid-19, or no illness. Using the Keras module, you will create a classification model that outputs a diagnosis based on a patient’s X-ray scan. You hope this model can help doctors with the challenge of deciphering X-ray scans and open a dialogue between your research team and the medical staff to create learning models that are as effective and interpretable as possible.

## Project Requirements
1. If you take a look at [Covid19 dataset](https://www.kaggle.com/pranavraikokte/covid19-image-dataset), you’ll see that there are two different folders inside. Take some time to look around these directories and familiarize yourself with the images inside. As you peruse through them, think about the following:

* What types of images are we working with?
* How are they organized? How will this affect preprocessing?
* Will this be binary classification or multi-class classification?

After you do this, you will be ready to get started on preprocessing! Click the hint below if you want to see some insights on the image data.


2. Load in your image data and get it ready for the journey through a neural network. One possible way to do this is to use an `ImageGenerator` object; however, feel free to try other methods you may have experienced before this project.

When creating the object, remember that neural networks struggle with large integer values. Think about how you might want to get your image data ready for your neural network and get the best results.


3. Now that you have set up your `ImageDataGenerator` object, it is time to actually load in your image data.

You will want to create two different iterable objects from this `ImageDataGenerator`: a train set and a test/validation set.

When you are creating these sets of images consider the following:

* The directory the images come from
* The types of images you are working with
* The target size of the images
* Click the hint below if you need any other guidance.

4. Now that your image data is loaded and ready for analysis, create a classification neural network model to perform on the medical data.

With image data, there are a ton of directions to go in. To get you grounded, we recommend you start by creating your input layer and output layer and compile it before diving into creating a more complex model.

When starting your neural network, consider the following:
* The shape of your input
* The shape of your output
* Using any activation functions for your output
* Your gradient descent optimizer
* Your learning rate
* Your loss functions and metrics
* Flattening the image data before the output layer

5. It’s time to test out the model you created!

Fit your model with your training set and test it out with your validation/test set.

Since you have not added many layers yet or played around with hyperparameters, it may not be very accurate yet. Do not fret! Your next adventure will be to play with your model and mold it until you see more ideal results.


6. You have created a model and tested it out. Now it is time for the real fun! Start playing around with some hidden layers and hyperparameters.

When adding hidden layers, consider the type of hidden layers you add (remember this is image data).

As you add in layers, you should also adjust your model’s hyperparameters. You have a lot of choices to make. You can choose:

* the number of `epochs`
* The size of your `batch_size`
* to add more hidden layers
* your type of optimizer and/or activation functions
* the size of your learning rate

Have fun in the hyperparameter playground. Test things out and see what works and what does not work. See what makes your model optimized between speed and accuracy. You have complete creative power!




Extensions
7. Plot the cross-entropy loss for both the train and validation data over each epoch using the Matplotlib Library. You can also plot the AUC metric for both your train and validation data as well. This will give you an insight into how the model performs better over time and can also help you figure out better ways to tune your hyperparameters.

Because of the way Matplotlib plots are displayed in the learning environment, please use `fig.savefig('static/images/my_plots.png')` at the end of your graphing code to render the plot in the browser. If you wish to display multiple plots, you can use `.subplot()` or `.add_subplot() `methods in the Matplotlib library to depict multiple plots in one figure.


***One way to plot your metrics is with the following code:***
```python
# plotting categorical and validation accuracy over epochs
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')

# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')

# used to keep plots from overlapping
fig.tight_layout()

fig.savefig('static/images/my_plots.png')
```


8. Another potential extension to this project would be implementing a [classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) and a [confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix). These are not tools we have introduced you to; however, if you would like further resources to improve your neural network, we recommend looking into these concepts.

As a brief introduction, these concepts evaluate the nature of false positives and false negatives in your neural network taking steps beyond simple evaluation metrics like accuracy.

In the hint below, you will see a possible solution to calculate a `classification_report` and a `confusion_matrix`, but you will need to do some personal googling/exploring to acquaint yourself with these metrics and understand the outputs.