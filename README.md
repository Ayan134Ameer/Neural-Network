# Neural Network

A series of neural networks built from scratch in Python/NumPy — no high-level ML frameworks (no TensorFlow/PyTorch/Keras) — written to build a real understanding of forward and backward propagation from the ground up, starting with small toy problems and working up to digit classification.

## Contents

### `XOR/`
A minimal neural network trained to learn the XOR function. First working proof of forward and backward propagation on a small, well-understood, non-linearly-separable problem.

### `3BitParity/`
Extends the same approach to a harder problem — predicting the parity (even/odd number of 1s) of a 3-bit input.

### `DigitRecog/`
A 64→32→16→10 fully-connected network trained from scratch on scikit-learn's `load_digits()` dataset, reaching **89% test accuracy**. Includes hand-derived cross-entropy and ReLU gradients for backpropagation.

### `DigitDrawingDemo/`
An interactive browser-based demo (HTML/JS) that loads the trained weights from `digit-classifier/` and classifies digits drawn freehand on a canvas.

**Note:** predictions are frequently wrong on freehand input, despite the underlying model's 89% test accuracy. This comes down to a **train/test distribution mismatch**, not a bug in the network: `load_digits()` images are small (8×8), pre-centered, pre-scaled, and lightly anti-aliased, while a canvas drawing has different stroke thickness, off-center positioning, and inconsistent scale — with no preprocessing step in between to normalize the raw canvas input into that same clean format. Fixing this would mean adding a preprocessing pipeline (centering, scaling, thresholding) between the canvas and the network.

### `DigitRecogBatched/` — *in progress*
A vectorized (batched) rewrite of the same network architecture, trained on the full MNIST dataset instead of `load_digits()`. Not yet complete.

## Resources
- *Essential Math for AI* by Hala Nelson — helped build intuition for the forward pass and for linear regression
- *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron — occasional reference
