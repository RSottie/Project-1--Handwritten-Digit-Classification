# Handwritten Digits Recognizer Using Two-Layer Neural Network

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

This project implements a **basic two-layer neural network from scratch** to classify handwritten digits using the MNIST dataset. The goal was to understand the **math behind neural networks**, including forward/backward propagation, activations, loss functions, and gradient updates.

---

## Motivation

This project was designed to **deeply understand neural networks** by implementing all components manually, without high-level frameworks like TensorFlow or PyTorch.  

Key concepts learned:  

- Forward and backward propagation  
- ReLU activation and Softmax output  
- Cross-entropy loss  
- Gradient descent updates  
- Weight initialization  

---

## Dataset

**MNIST Handwritten Digits Dataset**  

- Training set: 42,000 images (with labels)  
- Test set: 28,000 images (no labels)  
- Each image: 28x28 grayscale pixels  

Files included:  

- `train.csv` – training images and labels  
- `test.csv` – test images  
- `submission.csv` – sample predictions  

---

## Data Exploration

The notebook includes:

- Visualizing 10x10 random digits from the training set  
- Checking first and last rows of training data  
- Normalizing pixel values to `[0, 1]`  
- Splitting data into **training** and **development** sets  

---

## Neural Network Architecture

**Two-Layer Feedforward Network**:

- **Input Layer:** 784 neurons (28×28 flattened pixels)  
- **Hidden Layer:** 128 neurons (configurable) with **ReLU activation**  
- **Output Layer:** 10 neurons with **Softmax** for class probabilities  

---

## Functions Implemented

- **Initialization:** `init_param`  
- **Activations:** `relu`, `softmax`  
- **Forward propagation:** `forward_prop`  
- **Backward propagation:** `back_prop`  
- **Parameter updates:** `param_update`  
- **Prediction:** `predict`  
- **Accuracy:** `get_accuracy`  
- **Gradient descent training:** `gradient_descent`  

---

## Model Training

- Hyperparameters:

  ```text
  Learning rate: 0.1
  Hidden layer size: 128
  Iterations: 500
