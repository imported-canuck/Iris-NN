# Store one hot encoding and cross-entropy loss functions

import numpy as np

def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray: # Convert labels to one-hot encoding
    m = labels.shape[0]                    # One-hot encoding is a binary matrix representation of labels
    onehot = np.zeros((m, num_classes))    # For each label, a row of zeroes with a 1 at the index of the label
    onehot[np.arange(m), labels] = 1
    return onehot

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float: # Compute cross-entropy loss between true labels and predicted probabilities
    # y_true, y_pred both shape (m, C)
    eps = 1e-9 # Small value to avoid log(0)
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1)) # Average loss over all samples
