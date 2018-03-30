# Densenet_SVHN
A tensorflow implementation of dense net on Street view house number(SVHN) dataset.

## Model

    A brief description of the Model is provided below.

### High level view

Input layer -> Block 1 -> Transition 1 -> Block 2 -> Transition 2 -> Block 3 -> Batch Normalization -> Relu -> Global average pooling -> Fully connected layer

* **Block** consists of 4 Dense layers.

* **Dense layer** is made of the following sequence:
   1. Batch Normalization
   2. Relu
   3. Convolutional 2d layer
   4. Concatination of the previous layers output to the previous element(Convolutional 2d layer)

* **Transition Layer** is made of the following sequence:
   1. Batch Normalization
   2. Relu
   3. Convolutional 2d layer
   4. Average Pooling(stride=2)

### Hyper parameters and other essential attributes

* Input dimension = [100, 32, 32, 1] (Trained using batches of 100 images)
* Ouput dimension = [10]
* epoch = 800

### Preprocessing

* Balanced subsampling on training dataset.
* Converting SVHN images from RGB to grayscale.
* The training and validation data are stored in HDF5 binary data format.
