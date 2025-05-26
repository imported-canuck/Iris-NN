# Miscellaneous utility functions for training and evaluation

import numpy as np

def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float: # Choose the class with the highest probability as the predicted class
    # y_pred: probabilities shape (m, C) 
    labels = np.argmax(y_pred, axis=1)
    return np.mean(labels == y_true) # Check if predicted class matches the true class

def get_mini_batches(X, y, batch_size=16, shuffle=True): # Splits input data into mini-batches for training
    m = X.shape[0]     # Number of samples (rows)
    idx = np.arange(m) # Create an array of indices from 0 to m-1
    if shuffle:
        np.random.shuffle(idx) # Shuffle rows of the design matrix to ensure random sampling
    for start in range(0, m, batch_size): # Create mini-batches of size batch_size
        end = start + batch_size          # Mini-batches are more memory-efficient
        batch_idx = idx[start:end]
        yield X[batch_idx], y[batch_idx]
