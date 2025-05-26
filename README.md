# Iris Neural Network Classifier (from Scratch)
This repository contains a manually implemented feed-forward neural network classifier built entirely from scratch using Python. The neural network classifies flowers from the Iris dataset into their correct species, based on their given sepal and petal dimensions.  

No high-level libraries (e.g., TensorFlow, Keras, or Scikit-learn) are used to build or train the neural network. Instead, all functionality, from initial parameter setup, forward propagation, backward propagation, gradient computation, and stochastic gradient descent (SGD), is explicitly coded using fundamental mathematical concepts and NumPy for numerical computations, down to the individual matrix multiplications.  

Scikit-learn is used only for data loading to conveniently access the built-in Iris dataset, to avoid manual preprocessing.  

## Model Architecture
- **Input Layer**: 4 neurons, each corresponding to one of the Iris flower's measured features, all expressed as a float:
  -   Sepal length
  -   Sepal width
  -   Petal length
  -   Petal width
  
- **Hidden layers:** Two layers, each consisting of 16 neurons:
  - Each hidden neuron takes a weighted sum of inputs from the previous layer and passes it through a sigmoid activation function.
  - The hidden layers allow the network to learn complex nonlinear relationships between input features and species classification.

- **Output layer:** 3 neurons, corresponding to the three Iris flower species:
  - Setosa
  - Versicolor
  - Virginica

- **Activation functions:**
  - **Sigmoid** for hidden layers
  - **Softmax** for the output layer

- **Loss function:** Cross-entropy loss to measure how accurately the model predicts the correct species.

- **Optimization:** Mini-batch Stochastic Gradient Descent (SGD) for iterative training. Stochastic due to the randomization of the batches.

## How the Neural Network Works 

When presented with measurements of an Iris flower:

1. **Input Processing:**
   - The network takes the four measurements as inputs.

2. **Hidden Layers:**
   - Each hidden layer neuron calculates a weighted combination of the inputs (or previous layer outputs).
   - These combined values are then "squashed" into a range between 0 and 1 using a sigmoid activation function. This prevents explosions.
   - The first hidden layers identify basic patterns (ie, broad relationship between sepal length and class), while deeper layers attempt to capture complex features, often intangible to humans. 

4. **Output Layer:**
   - The final layer computes another weighted combination from the penultimate hidden layer’s outputs.
   - The result is passed through a softmax function, producing multivariate probabilities for each flower species.
   - The species with the highest probability is chosen as the predicted species.

5. **Learning:**
   - Initially, the network guesses randomly, as weights are set to small, random values (Kaiming initialization, might change to Xavier/Glorot later).
   - Each guess is compared to the correct species using cross-entropy loss to measure the prediction's "wrongness" from the ground truth.
   - The network then uses back-propagation, a mathematical process, to determine how each weight and bias contributed to the error.
   - Weights and biases are adjusted slightly after each batch of examples to reduce the error. Over many iterations (epochs), the network gradually improves its predictions.
  
## Dependencies

- Python (NumPy)
- Scikit-learn (only for data loading)

No external neural-network packages are required (that's the point, duh).

## Usage

To run the network, simply execute:

```bash
python main.py
```

Feel free to play around with the hyperparameters, activations, or adapt the neural network to a new dataset.  

## Project Structure
```bash
.
├── data_loader.py          # Loads Iris dataset
├── neural_network.py       # Neural network logic (forward/backward passes)
├── activations.py          # Activation functions
├── loss.py                 # Loss function computations
├── trainer.py              # Training loop with SGD
├── utils.py                # Helper functions (accuracy, batching)
└── main.py                 # Entry point script
```

## License 
MIT License. See the [LICENSE](LICENSE) file for details.
