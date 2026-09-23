# Home Assignment 2: Neural Networks and Deep Learning

`Name: Aryaman Shrestha`

`Student ID: 700788013`

--- 

## Contents

| Notebook | Topic |
| --- | --- |
| [Question1.ipynb](Question1.ipynb) | Character-level Shakespeare text generation with an LSTM |
| [Question2.ipynb](Question2.ipynb) | IMDB movie-review sentiment classification with an LSTM |
| [Question3.ipynb](Question3.ipynb) | 2D convolution with different strides and padding modes |
| [Question4.ipynb](Question4.ipynb) | Sobel edge detection and max/average pooling |
| [Question5.ipynb](Question5.ipynb) | Simplified AlexNet and ResNet-like model architectures |


## Questions

### Question 1: Character-Level Text Generation

[Question1.ipynb](Question1.ipynb) trains an LSTM to generate Shakespeare-like text one character at a time.

- Downloads the Shakespeare text dataset from TensorFlow storage.
- Uses sequences of 100 input characters and 100 target characters.
- Encodes 65 unique characters as integer IDs and one-hot vectors.
- Uses a 256-unit LSTM followed by a 65-unit output layer.
- Trains with Adam and categorical cross-entropy for 50 epochs.
- Generates text from the prompt `First Citizen:` using temperature sampling.

The notebook uses batches shaped `(64, 100, 65)` and produces generated text whose randomness can be adjusted with the temperature value. Internet access is required on the first run, and training may take significant time.

### Question 2: IMDB Sentiment Classification

[Question2.ipynb](Question2.ipynb) classifies movie reviews as positive or negative.

- Uses the Keras IMDB dataset with 25,000 training and 25,000 test reviews.
- Restricts the vocabulary to the 10,000 most frequent words.
- Pads or truncates each review to 200 tokens.
- Uses a 128-dimensional embedding, a 64-unit LSTM, and a sigmoid output.
- Trains with binary cross-entropy, Adam, and a 20% validation split.
- Reports test accuracy, a confusion matrix, and a classification report.

The stored run achieves approximately 82% test accuracy. The notebook also discusses how changing the classification threshold affects precision and recall. The IMDB dataset may be downloaded automatically by Keras on the first run.

### Question 3: Convolution, Stride, and Padding

[Question3.ipynb](Question3.ipynb) applies a Laplacian-style `3 x 3` kernel to a fixed `5 x 5` matrix with TensorFlow `tf.nn.conv2d`.

It compares these configurations:

- Stride 1 with `VALID` padding
- Stride 1 with `SAME` padding
- Stride 2 with `VALID` padding
- Stride 2 with `SAME` padding

This notebook is self-contained and does not require a dataset or image. The `VALID` configurations produce smaller outputs, while `SAME` preserves the spatial dimensions according to the stride.

### Question 4: Sobel Filters and Pooling

[Question4.ipynb](Question4.ipynb) contains two image-processing exercises.

**Sobel edge detection**

- Reads a grayscale image with OpenCV.
- Applies horizontal and vertical Sobel kernels.
- Displays the original image and both edge maps.

The Sobel section expects an image named `lena.jpg` in the notebook's working directory. That file is not included in this repository, so add it before running the image-filtering cells.

**Pooling**

- Creates a deterministic `4 x 4` matrix using NumPy seed 42.
- Applies `2 x 2` max pooling and average pooling with Keras.
- Produces `2 x 2` pooled outputs.

The pooling section does not require `lena.jpg`.

### Question 5: AlexNet and ResNet-Like Architectures

[Question5.ipynb](Question5.ipynb) builds, compiles, and summarizes two classification models. It does not train either model.

**Simplified AlexNet**

- Accepts input shaped `(227, 227, 3)`.
- Uses convolution, max-pooling, 4096-unit dense layers, and dropout.
- Produces 10 softmax class probabilities.

**ResNet-like model**

- Accepts input shaped `(64, 64, 3)`.
- Uses an initial convolution followed by two residual blocks.
- Adds skip connections with element-wise addition.
- Produces 10 softmax class probabilities.

The notebook prints the model summaries and parameter counts. No dataset or external image is required.

## Notes

- Run notebooks from the repository directory so relative paths, including `lena.jpg`, resolve correctly.
- Questions 1 and 2 download datasets or source text and therefore require internet access on a fresh environment.
- Notebook outputs may vary slightly with TensorFlow, hardware, and random initialization.
