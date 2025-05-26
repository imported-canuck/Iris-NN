# Store activation functions and their derivatives

import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray: # Sigmoid(x) = 0 for x < 0, 1 for x > 0
    return 1.0 / (1.0 + np.exp(-z))       # Acts as a soft yes/no (0/1) gate, but still changes gradually
                                          # Gradual change answers "how strongly yes or no"
def sigmoid_derivative(a: np.ndarray) -> np.ndarray: # Derivative of sigmoid function
    # a = sigmoid(z)                                 # σ'(z) = σ(z)(1 - σ(z))
    return a * (1 - a)                               # Clean derivative for backpropagation

def softmax(z: np.ndarray) -> np.ndarray: # Softmax generalizes sigmoid to multiple classes
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) # Inputs an array of logits (raw scores)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)  # Outputs a vector of class probabilities that sum to 1 
