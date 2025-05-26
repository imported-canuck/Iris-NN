from loss import one_hot, cross_entropy
from utils import accuracy, get_mini_batches

class Trainer:
    def __init__(self, model, X_train, y_train, X_val, y_val):
        self.model = model # An instance of NeuralNetwork
        self.X_train, self.y_train = X_train, y_train # Training data
        self.X_val,   self.y_val   = X_val,   y_val   # Validation data

    def train(self, epochs=50, batch_size=16, lr=0.01): 
        for epoch in range(1, epochs+1): # Epochs are the number of times the model sweeps through the dataset
            # Mini-batch SGD
            for X_batch, y_batch in get_mini_batches(self.X_train, self.y_train, batch_size): # Split data into mini batches
                y_batch_oh = one_hot(y_batch, self.model.sizes[-1])   # Convert labels to one-hot 
                acts, zs = self.model.forward(X_batch)                # Forward pass one that batch
                dW, db = self.model.backward(acts, zs, y_batch_oh)    # Backward pass to compute gradients
                self.model.update_params(dW, db, lr)                  # Update model parameters accoprding to gradients

            # Evaluate (ignore zs)
            train_acts, _ = self.model.forward(self.X_train)          # Reconstruct activations for training data
            val_acts, _   = self.model.forward(self.X_val)            # Reconstruct activations for validation data   
            train_loss = cross_entropy(one_hot(self.y_train, self.model.sizes[-1]), train_acts[-1]) # Compute training loss
            train_acc  = accuracy(train_acts[-1], self.y_train)     # Compute training accuracy
            val_acc    = accuracy(val_acts[-1], self.y_val)         # Compute validation accuracy

            print(f"Epoch {epoch:>2} | loss: {train_loss:.4f} | train_acc: {train_acc:.3f} | val_acc: {val_acc:.3f}")
