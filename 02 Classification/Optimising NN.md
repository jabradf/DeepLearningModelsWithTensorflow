# Optimising Neural Networks
From the entry on [Stack Exchange](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw):

By following a small set of clear rules, one can programmatically set a competent network architecture (i.e., the number and type of neuronal layers and the number of neurons comprising each layer). Following this schema will give you a competent architecture but probably not an optimal one.

Once this network is initialized, you can iteratively tune the configuration during training using a number of ancillary algorithms; one family of these works by pruning nodes based on (small) values of the weight vector after a certain number of training epochs--in other words, eliminating unnecessary/redundant nodes (more on this below).

So every NN has three types of layers: _input_, _hidden_, and _output_.

Creating the NN architecture, therefore, means coming up with values for the number of layers of each type and the number of nodes in each of these layers.

## The Input Layer

Simple--every NN has exactly one of them--no exceptions that I'm aware of.

With respect to the number of neurons comprising this layer, this parameter is completely and uniquely determined once you know the shape of your training data. Specifically, `the number of neurons comprising that layer is equal to the number of features (columns) in your data`. Some NN configurations add one additional node for a bias term.

# The Output Layer

Like the Input layer, every NN has exactly one output layer. Determining its size (number of neurons) is simple; `it is completely determined by the chosen model configuration`.

Is your NN going to run in Machine Mode or Regression Mode (the ML convention of using a term that is also used in statistics but assigning a different meaning to it is very confusing)? Machine mode: returns a class label (e.g., "Premium Account"/"Basic Account"). Regression Mode returns a value (e.g., price).

***If the NN is a regressor, then the output layer has a single node.***

If the NN is a classifier, then it also has a single node unless softmax is used in which case the output layer has one node per class label in your model.

## The Hidden Layers

So those few rules set the number of layers and size (neurons/layer) for both the input and output layers. That leaves the hidden layers.

How many hidden layers? Well, if your data is linearly separable (which you often know by the time you begin coding a NN), then you don't need any hidden layers at all. Of course, you don't need an NN to resolve your data either, but it will still do the job.

Beyond that, as you probably know, there's a mountain of commentary on the question of hidden layer configuration in NNs (see the insanely thorough and insightful [NN FAQ](http://www.faqs.org/faqs/ai-faq/neural-nets/part1/preamble.html) for an [excellent summary](http://www.faqs.org/faqs/ai-faq/neural-nets/part1/preamble.html) of that commentary). One issue within this subject on which there is a consensus is the performance difference from adding additional hidden layers: the situations in which performance improves with a second (or third, etc.) hidden layer are very few. One hidden layer is sufficient for the large majority of problems.

So what about the size of the hidden layer(s)--how many neurons? There are some empirically derived rules of thumb; of these, the most commonly relied on is '`the optimal size of the hidden layer is usually between the size of the input and size of the output layers`'. Jeff Heaton, the author of [Introduction to Neural Networks in Java](https://www.heatonresearch.com/book/), offers a few more.

In sum, for most problems, one could probably get decent performance (even without a second optimization step) by setting the hidden layer configuration using just two rules: (i) the number of hidden layers equals one; and (ii) the number of neurons in that layer is the mean of the neurons in the input and output layers.

## Optimization of the Network Configuration

**Pruning** describes a set of techniques to trim network size (by nodes, not layers) to improve computational performance and sometimes resolution performance. The gist of these techniques is removing nodes from the network during training by identifying those nodes which, if removed from the network, would not noticeably affect network performance (i.e., resolution of the data). 

Even without using a formal pruning technique, you can get a rough idea of which nodes are not important by looking at your weight matrix after training; **look at weights very close to zero**--it's the nodes on either end of those weights that are often removed during pruning.

Obviously, if you use a pruning algorithm during training, then begin with a network configuration that is more likely to have excess (i.e., 'prunable') nodes--in other words, when deciding on network architecture, err on the side of more neurons, if you add a pruning step.

Put another way, by applying a pruning algorithm to your network during training, you can approach optimal network configuration; whether you can do that in a single "up-front" (such as a genetic-algorithm-based algorithm), I don't know, though I do know that for now, this two-step optimization is more common.