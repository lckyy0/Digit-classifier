# MNIST Digit Classifier from Scratch

A handwritten digit classifier built with only NumPy, pandas and matplotlib. No TensorFlow, PyTorch or scikit-learn is used for the model. Forward propagation, backpropagation and gradient descent are all written by hand, following my own derivation notes.

## Overview

- Input: 28 x 28 grayscale images, flattened to 784 pixels
- Network: 784 -> 256 (ReLU) -> 10 (softmax)
- Loss: cross-entropy
- Training: full-batch gradient descent
- Output: one probability per digit 0-9, and the prediction is the argmax

## Dataset

The data is the [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) `train.csv`: 42,000 labeled images, where the first column is the label and the other 784 columns are pixel values from 0 to 255.

The Kaggle `test.csv` has no labels, so it is not used. Instead, `train.csv` is shuffled (seed 42) and split into:

| Set | Images | Purpose |
|---|---|---|
| Train | 32,000 | Fit the weights |
| Validation | 5,000 | Tune alpha, iterations and hidden size |
| Test | 5,000 | Evaluated once at the very end |

The CSV is not included in this repository. Download it from Kaggle and place it at `data/train.csv`.

## Project structure

```
.
├── data/
│   └── train.csv        (download from Kaggle, not committed)
├── images/
│   ├── notes_forward_backward.png
│   └── notes_update_rule.png
├── mnist.ipynb
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put `train.csv` in the `data/` folder, open `mnist.ipynb` in VS Code or Jupyter, and run all cells from top to bottom.

Training on a CPU-only laptop takes roughly 9 minutes for 2000 iterations with 256 hidden neurons.

## Notebook walkthrough

| Step | What it does |
|---|---|
| 1 | Imports and random seed |
| 2 | Loads the CSV, shuffles, splits, scales pixels to 0-1, transposes to `(784, m)` and one-hot encodes labels to `(10, m)` |
| 3 | Plots sample digits |
| 4 | `init_params`, `relu`, `softmax`, `forward_prop` |
| 5 | `compute_loss`, `get_predictions`, `get_accuracy` |
| 6 | `backward_prop` |
| 7 | `update_params` |
| 8 | Numerical gradient check |
| 9 | Training function and training run, with logging every 50 iterations |
| 10 | Loss and accuracy curves, final train and validation accuracy |
| 11 | Test set evaluation, confusion matrix, misclassified digits |
| 12 | Saving and loading weights with `np.savez` / `np.load`, and predicting a single image |

## The math

Notation follows my handwritten notes in `images/`. Here `m` is the number of images and `n_h` is `hidden_size`.

![Forward and backward propagation notes](images/notes_forward_backward.png)

![Update rule notes](images/notes_update_rule.png)

### Forward propagation

$$A^{[0]} = X$$

$$Z^{[1]} = W^{[1]} A^{[0]} + b^{[1]}$$

$$A^{[1]} = \mathrm{ReLU}(Z^{[1]})$$

$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$

$$A^{[2]} = \mathrm{softmax}(Z^{[2]})$$

Softmax is computed per column, after subtracting the column maximum for numerical stability.

### Loss

$$J = -\frac{1}{m} \sum Y \log A^{[2]}$$

### Backward propagation

$$dZ^{[2]} = A^{[2]} - Y$$

$$dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}$$

$$db^{[2]} = \frac{1}{m} \sum dZ^{[2]}$$

$$dZ^{[1]} = W^{[2]T} dZ^{[2]} \odot g'(Z^{[1]}), \quad g'(Z^{[1]}) = (Z^{[1]} > 0)$$

$$dW^{[1]} = \frac{1}{m} dZ^{[1]} X^T$$

$$db^{[1]} = \frac{1}{m} \sum dZ^{[1]}$$

### Parameter update

$$W^{[l]} := W^{[l]} - \alpha \, dW^{[l]}, \qquad b^{[l]} := b^{[l]} - \alpha \, db^{[l]}$$

where `alpha` is the learning rate.

### Shapes

| Array | Shape |
|---|---|
| `X`, `A0` | (784, m) |
| `Y` | (10, m) |
| `W1` | (n_h, 784) |
| `b1` | (n_h, 1) |
| `Z1`, `A1` | (n_h, m) |
| `W2` | (10, n_h) |
| `b2` | (10, 1) |
| `Z2`, `A2` | (10, m) |

Each gradient has the same shape as the parameter it updates.

## Training details

| Setting | Value |
|---|---|
| `hidden_size` | 256 |
| `alpha` | 1.0 |
| `iterations` | 2000 |
| Weight init | `randn * sqrt(2 / n_inputs)` |
| Bias init | zeros |
| Data type | float32 |

### How the settings were chosen

These notes come from earlier experiments with a 128-neuron network on the same kind of split, judged on validation accuracy only:

- Over 200 iterations, `alpha` = 1.0 beat 0.5, 0.1 and 0.05. A smaller learning rate did not give a better fit, only slower progress.
- With `alpha` = 1.0 and 128 neurons, validation accuracy flattened out at around 97.4% to 97.5% by about 1000 iterations, while train accuracy kept rising.
- A 10-neuron network at `alpha` = 1.0 failed (about 11% validation accuracy) because almost all hidden neurons died: their `Z1` went negative for every image, so ReLU gave 0 and the gradient `(Z1 > 0)` was 0. At `alpha` = 0.1 the same network reached about 91% with no dead neurons. The best learning rate depends on network size.

The final configuration (256 neurons, 2000 iterations) was chosen to give the model more capacity and time to reach its plateau.

## Results

Fill these in after running the notebook.

| Metric | Value |
|---|---|
| Final train accuracy | |
| Final validation accuracy | |
| Test accuracy (5,000 images) | |
| Test loss | |

The test set is used once, after all settings are fixed.

## Common bugs to watch for

- Forgetting the `1/m` factor in the gradients, which makes them `m` times too large
- Softmax or sum over the wrong axis (softmax uses `axis=0`, the bias gradient sum uses `axis=1`)
- Shape mismatches from a missing transpose, so `X` must be `(784, m)`
- Biases of shape `(n,)` instead of `(n, 1)`
- Starting loss far from about 2.3 (`-log(0.1)`) for 10 classes, which points to a bug in initialization or the loss
- Not re-assigning the weights returned by `update_params`

## Possible improvements

- Mini-batch gradient descent
- Adam or momentum
- L2 regularization or dropout
- A second hidden layer
- Data augmentation, or a convolutional network

## Acknowledgements

Dataset: [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer), derived from the MNIST database by Yann LeCun et al.
