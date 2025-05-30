# Load the iris dataset and split it into training and testing sets.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(test_size=0.2, random_state=1): # Train-test split is 80-20
    iris = load_iris()
    X = iris.data.astype(np.float32)      # Design matrix (150, 4)
    # Each of the 4 columns is a feature of the data point. 
    y = iris.target                       # Labels (150,) 1D array
    # Shuffle & split
    return train_test_split(X, y,
                            test_size=test_size,
                            random_state=random_state,
                            stratify=y)   # stratify to preserve class distribution
