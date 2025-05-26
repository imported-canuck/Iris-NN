from data_loader import load_data
from neural_network import NeuralNetwork
from trainer import Trainer

def main():
    # Hyperparameters
    layers = [4, 16, 16, 3]    # Neural network architecture: 4 input features, 2 hidden layers with 16 neurons each, and 3 output classes
    epochs = 100               # Number of epochs to train
    batch  = 16                # Batch size for mini-batch gradient descent
    lr     = 0.05              # Learning rate for weight updates

    # Load Iris
    X_train, X_val, y_train, y_val = load_data(test_size=0.2)

    # Build & train
    model   = NeuralNetwork(layers)
    trainer = Trainer(model, X_train, y_train, X_val, y_val)
    trainer.train(epochs=epochs, batch_size=batch, lr=lr)

if __name__ == "__main__":
    main()
