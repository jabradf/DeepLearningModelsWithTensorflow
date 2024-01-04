IMAGE CLASSIFICATION
# A Better Alternative: Convolutional Neural Networks
Convolutional Neural Networks (CNNs) use layers specifically designed for image data. These layers capture local relationships between nearby features in an image.

Previously, in our feed-forward model, we multiplied our normalized pixels by a large weight matrix (of shape `(65536, 100)`) to generate our next set of features.

However, when we use a convolutional layer, we learn a set of smaller weight tensors, called filters (also known as kernels). We move each of these filters (i.e. convolve them) across the height and width of our input, to generate a new “image” of features. Each new “pixel” results from applying the filter to that location in the original image.

The interactive on the right demonstrates how we convolve a filter across a single image.

## Why do convolution-based approaches work well for image data?
* Convolution can reduce the size of an input image using only a few parameters.
* Filters compute new features by only combining features that are near each other in the image. This operation encourages the model to look for local patterns (e.g., edges and objects).
* Convolutional layers will produce similar outputs even when the objects in an image are translated (For example, if there were a giraffe in the bottom or top of the frame). This is because the same filters are applied across the entire image.

Before deep nets, researchers in computer vision would hand design these filters to capture specific information. For example, a 3x3 filter could be hard-coded to activate when convolved over pixels along a vertical or horizontal edge: 

![Edge Filters](img/Edge%20Detector%20Filters.png)

*The image displays two filters. One is a vertical edge detector, with zeros in the center column, ones in the left column, and negative ones in the right column. The other is a horizontal edge detector, with ones in the top row, negative ones in the bottom row, and zeros in the middle row.*

However, with deep nets, we can learn each weight in our filter (along with a bias term)! As in our feed-forward layers, these weights are learnable parameters. Typically, we randomly initialize our filters and use gradient descent to learn a better set of weights. By randomly initializing our filters, we ensure that different filters learn different types of information, like vertical versus horizontal edge detectors.