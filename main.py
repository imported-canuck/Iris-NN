from data_loader import load_data
from neural_network import NeuralNetwork
from trainer import Trainer

def main():
    # Hyperparameters
    layers = [64, 32, 32, 10]  # Neural network architecture: 64 input features (pixels), 2 hidden layers with 32 neurons each, and 10 output classes
    epochs = 500               # Number of epochs to train
    batch  = 32                # Batch size for mini-batch gradient descent
    lr     = 0.05              # Learning rate for weight updates

    # Load Iris
    X_train, X_val, y_train, y_val = load_data(test_size=0.2)

    # Build & train
    model   = NeuralNetwork(layers)
    trainer = Trainer(model, X_train, y_train, X_val, y_val)
    trainer.train(epochs=epochs, batch_size=batch, lr=lr)

if __name__ == "__main__":
    main()
