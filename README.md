# Handwritten Digits Recognizer Using Two-Layer Neural Network

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

This project implements a **basic two-layer neural network from scratch** to classify handwritten digits using the MNIST dataset. The goal was to understand the **mathematics behind neural networks**, including forward and backward propagation, activation functions, loss computation, and gradient-based optimization.

---

## Motivation

This project was created to gain a **deep, first-principles understanding of neural networks** by implementing all components manually, without using high-level frameworks such as TensorFlow or PyTorch.

Key concepts learned include:
- Forward and backward propagation
- ReLU activation and Softmax output
- Cross-entropy loss
- Gradient descent optimization
- Weight and bias initialization

---

## Dataset

**MNIST Handwritten Digits Dataset**

- Training set: 42,000 labeled images  
- Test set: 28,000 unlabeled images  
- Image size: 28×28 grayscale pixels  

Included files:
- `train.csv` – training images and labels
- `test.csv` – test images
- `submission.csv` – sample predictions

---

## Data Exploration

The notebook covers:
- Visualization of random handwritten digits
- Inspection of dataset structure
- Normalization of pixel values to `[0, 1]`
- Splitting the dataset into training and development sets

---

## Neural Network Architecture

**Two-Layer Feedforward Neural Network**

- Input Layer: 784 neurons (flattened 28×28 pixels)
- Hidden Layer: 128 neurons with ReLU activation
- Output Layer: 10 neurons with Softmax activation

---

## Functions Implemented

- `init_param` – parameter initialization
- `relu`, `softmax` – activation functions
- `forward_prop` – forward propagation
- `back_prop` – backward propagation
- `param_update` – gradient descent updates
- `predict` – digit prediction
- `get_accuracy` – accuracy computation
- `gradient_descent` – training loop

---

## Model Training

Hyperparameters:
```
Learning rate: 0.1
Hidden layer size: 128
Iterations: 500
```

Results:
- Final Training Accuracy: 92.0%
- Development Accuracy: 92.0%

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/RSottie/Project-1--Handwritten-Digit-Classification.git
cd Project-1--Handwritten-Digit-Classification
```

2. Install dependencies:
```bash
pip install numpy pandas matplotlib jupyter
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open `Neural_Network_From_Scratch.ipynb` and run all cells.

---

## Test Set Predictions

The notebook includes utilities to visualize predictions on the test dataset. Ground-truth labels are not provided for the test set.

---

## Future Improvements

- Add additional hidden layers
- Implement advanced optimizers (Momentum, Adam)
- Apply regularization techniques (Dropout, L2)
- Visualize learned weight patterns as digit templates

---

## License

This project is licensed under the MIT License.
