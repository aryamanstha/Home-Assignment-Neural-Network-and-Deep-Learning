# Neural Network and Deep Learning Assignment 3

### Name: Aryaman Shrestha

### Student ID: 700788013

--------------------------
## Contents

### Question 1: Convolution from scratch

[Question1.ipynb](./Question1.ipynb) implements 2-D filtering without calling
a built-in convolution function.

The notebook:

1. Stores the provided 5 x 5 input matrix and 3 x 3 filter as NumPy arrays.
2. Defines `convolve_2d`, which calculates the output dimensions from the
   input shape, kernel shape, and stride.
3. Slides the filter across every valid input region.
4. Computes one output value at each position by multiplying corresponding
   values and summing them.
5. Prints the feature map and its shape for stride 1.
6. Repeats the calculation with stride 2 to show how skipping positions
   reduces the output size and the number of computations.

With stride 1 and no padding, the 5 x 5 input produces a 3 x 3 feature map.
With stride 2, it produces a 2 x 2 feature map. A larger stride is faster
because it evaluates fewer positions, but it also retains less spatial detail.

### Question 2: Transfer learning

[Question2.ipynb](./Question2.ipynb) uses a pretrained ResNet18 classifier with
the CIFAR-10 image-classification dataset. The notebook applies the
pretrained model's default image transformations, selects 500 training images
from each of CIFAR-10's 10 classes, and uses the same selected images and
random seed for the experiments.

#### Experiment A: Frozen feature extractor

- Loads pretrained ResNet18 weights.
- Freezes all existing model parameters.
- Replaces the final fully connected layer with a 10-class classifier.
- Trains only the replacement classifier.
- Reports trainable parameters, training time, and test accuracy.
- Plots training loss over the epochs.

#### Experiment B: Fine-tuned network

- Starts from the same pretrained ResNet18 architecture and classifier setup.
- Keeps the classifier trainable.
- Also unfreezes `layer4`, the last convolutional block.
- Uses a smaller learning rate for the fine-tuned convolutional block.
- Reports the same metrics and plots both loss curves for comparison.

The notebook supports CUDA, Apple Metal Performance Shaders (MPS), and CPU
execution, choosing the first available device. It also synchronizes GPU
operations before measuring elapsed time so that the reported training times
are meaningful.

The recorded comparison is:

| Method | Trainable parameters | Training time | Test accuracy |
| :--- | ---: | ---: | ---: |
| Frozen feature extractor | 5,130 | 51.53 seconds | 82.97% |
| Fine-tuned network | 8,398,858 | 71.56 seconds | 88.00% |

The frozen approach is faster because only the final classifier receives
gradients and updates. Fine-tuning requires additional backpropagation through
the last convolutional block, but it can adapt pretrained visual features to
the target dataset and therefore achieve better accuracy. Fine-tuning can also
overfit when the training set is small.

