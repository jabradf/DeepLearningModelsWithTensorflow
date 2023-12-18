'''
The batch size is a hyperparameter that determines how many training samples are seen before updating the 
network’s parameters (weight and bias matrices).

When the batch contains all the training examples, the process is called batch gradient descent. 
If the batch has one sample, it is called the stochastic gradient descent. And finally, when 1 < batch size < number
of training points, is called mini-batch gradient descent. An advantage of using batches is for GPU computation that 
can parallelize neural network computations.

How do we choose the batch size for our model? On one hand, a larger batch size will provide our model with better 
gradient estimates and a solution close to the optimum, but this comes at a cost of computational efficiency and 
good generalization performance. On the other hand, smaller batch size is a poor estimate of the gradient, but 
the learning is performed faster. Finding the “sweet spot” depends on the dataset and the problem, and can be 
determined through hyperparameter tuning.

For this experiment, we fix the learning rate to 0.01 and try the following batch sizes: 1, 2, 10, and 16. Notice 
how small batch sizes have a larger variance (more oscillation in the learning curve).

Want to improve the performance with a larger batch size? A good trick is to increase the learning rate!
'''

from model import features_train, labels_train, design_model
import matplotlib.pyplot as plt

def fit_model(f_train, l_train, learning_rate, num_epochs, batch_size, ax):
    model = design_model(features_train, learning_rate)
    #train the model on the training data
    history = model.fit(features_train, labels_train, epochs=num_epochs, batch_size = batch_size, verbose=0, validation_split = 0.3)
    # plot learning curves
    ax.plot(history.history['mae'], label='train')
    ax.plot(history.history['val_mae'], label='validation')
    ax.set_title('batch = ' + str(batch_size), fontdict={'fontsize': 8, 'fontweight': 'medium'})
    ax.set_xlabel('# epochs')
    ax.set_ylabel('mae')
    ax.legend()

#fixed learning rate 
#learning_rate = 0.01 
learning_rate = 0.1 # improved rates here
#fixed number of epochs
num_epochs = 100
#we choose a number of batch sizes to try out
#batches = [2, 10, 16] 
batches = [4, 32, 64] 
print("Learning rate fixed to:", learning_rate)

#plotting code
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.7, 'wspace': 0.4}) #preparing axes for plotting
axes = [ax1, ax2, ax3]

#iterate through all the batch values
for i in range(len(batches)):
  fit_model(features_train, labels_train, learning_rate, num_epochs, batches[i], axes[i])

plt.show()
#plt.savefig('static/images/my_plot.png')
#print("See the plot on the right with batch sizes", batches)
#import app #don't worry about this. This is to show you the plot in the browser.