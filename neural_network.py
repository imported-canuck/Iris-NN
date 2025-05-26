import numpy as np
from activations import sigmoid, sigmoid_derivative, softmax

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: list of ints, e.g. [4, 10, 10, 3]
        """
        self.num_layers = len(layer_sizes) - 1 # -1 because first layer is input (no weights)
        self.sizes = layer_sizes               
        rng = np.random.default_rng(1)         # Seeded so every run starts with the same weights
        # He initialization for hidden layers (prevent vanishing or exploding gradients)
        self.weights = [
            rng.standard_normal((layer_sizes[i], layer_sizes[i+1])) * np.sqrt(2/layer_sizes[i])
            for i in range(self.num_layers)
        ]
        self.biases = [                        # Biases initialized to zero
            np.zeros((1, layer_sizes[i+1]))
            for i in range(self.num_layers)
        ]

    def forward(self, X):
        """
        Returns lists of activations and pre-activations:
          activations: [a0, a1, ..., aL]
          zs: pre-activations [z1, ..., zL]
        """
        a = X
        activations = [a]
        zs = []
        for i in range(self.num_layers):              # For all layers except output layer
            z = a @ self.weights[i] + self.biases[i]  # Z = A @ W + b
            zs.append(z)
            if i == self.num_layers - 1:              # If last layer, use softmax       
                # output layer
                a = softmax(z)
            else:                                     # If hidden layer, use sigmoid
                a = sigmoid(z)
            activations.append(a)
        return activations, zs # Activations is an array of the output matrices of each layer (the A = sigmoid(Z) or softmax(Z) matrix)

    def backward(self, activations, zs, y_true):
        """
        Compute gradients over one batch.
        y_true: one-hot encoded (batch_size, C)
        """
        m = y_true.shape[0]                           # Number of samples in batch
        # init gradient lists
        dW = [np.zeros_like(W) for W in self.weights] # Make empty matrix of same shape as each weight matrix
        db = [np.zeros_like(b) for b in self.biases]  # Make empty array of same shape as each bias vector

        # output layer error
        delta = activations[-1] - y_true                  # The softmax probabilities assigned to each class minus the one-hot encoded true labels
        # delta is the error for the output layer, positive entries indicate overestimation, negative entries indicate underestimation
        dW[-1] = activations[-2].T @ delta / m            # Determine the gradient of the weights for the outpit layer adjustmment
        db[-1] = np.sum(delta, axis=0, keepdims=True) / m # Average the error over the batch to get the gradient of the biases for the output layer
        # backpropagate through hidden layers
        for l in range(2, self.num_layers+1):             # count 2...L, grabbing layers from the end
            z = zs[-l]                                    # Get the pre-activation of the current layer                                 
            sp = sigmoid_derivative(activations[-l])      # Get the derivative of the activation function for the current layer
            delta = (delta @ self.weights[-l+1].T) * sp   # Backpropagate the error to the previous layer (compute next delta)
            dW[-l] = activations[-l-1].T @ delta / m      # Compute the gradient of the weights for next current layer (like was "initialized" above)
            db[-l] = np.sum(delta, axis=0, keepdims=True) / m # Compute the gradient of the biases for next current layer

        return dW, db

    def update_params(self, dW, db, lr):
        for i in range(self.num_layers):   # SGD step, update weights and biases in the direction of the gradient
            self.weights[i] -= lr * dW[i]  
            self.biases[i]  -= lr * db[i]
