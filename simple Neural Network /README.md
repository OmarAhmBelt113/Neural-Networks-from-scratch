# Neural Network Implementation

This repository contains an implementation of a simple feedforward neural network with one hidden layer. The network is designed to solve a basic regression problem using the sigmoid activation function and backpropagation for training.

## Table of Contents
- Overview
- Network Architecture
- Files
- Dependencies
- Usage
- Visualization
- Contributing

## Overview
This neural network is a two-layer feedforward network with the following characteristics:

- **Input Layer** : 2 neurons (I1, I2).
- **Hidden Layer** : 2 neurons (H1, H2) with biases.
- **Output Layer** : 2 neurons (O1, O2) with biases.
- **Activation Function** : Sigmoid function for both hidden and output layers.
- **Learning Algorithm** : Backpropagation with gradient descent.

The network is trained to minimize the mean squared error (MSE) between the predicted outputs and the target values.

## Network Architecture
### Structure
The neural network has the following structure:
```
Input Layer (I1, I2) → Hidden Layer (H1, H2) → Output Layer (O1, O2)
        ↑ Bias (B1)               ↑ Bias (B2)
```
- **Input Layer** : Receives input data.
- **Hidden Layer** : Processes the input using weights and biases, applying the sigmoid activation function.
- **Output Layer** : Produces the final predictions using weights and biases, applying the sigmoid activation function.

### Weights and Biases
- Each connection between layers has an associated weight.
- Each hidden and output neuron has a bias term.

## Files
- **`main.py`** : The main script that initializes the network, performs forward propagation, calculates errors, and updates weights using backpropagation.
- **`neural_network_module.py`** : A module containing reusable functions for forward propagation, error calculation, backpropagation, and weight updates.
- **`plot_neural_network_with_bias.py`** : A script to visualize the neural network structure, including biases, using matplotlib and networkx.
- **`README.md`** : This file, providing an overview of the project and instructions for use.

## Dependencies
To run the code, you need the following Python libraries:

- `numpy`: For numerical computations.
- `matplotlib`: For plotting the neural network visualization.
- `networkx`: For creating and visualizing the graph structure of the neural network.

You can install these dependencies using pip:
```bash
pip install numpy matplotlib networkx
```

## Usage
### Step 1: Run the Neural Network
To execute the neural network training:

1. Open a terminal or command prompt.
2. Navigate to the directory containing the files.
3. Run the main script:
   ```bash
   python main.py
   ```
   This will perform one iteration of training and print the updated weights.

### Step 2: Visualize the Neural Network
To visualize the structure of the neural network, including biases:

Run the visualization script:
```bash
python plot_neural_network_with_bias.py
```
This will generate a diagram showing the connections between layers and the biases.

## Visualization
The neural network structure is visualized as a directed graph, where:

- **Nodes** represent neurons in the input, hidden, and output layers.
- **Edges** represent weighted connections between neurons.
- **Bias Nodes (B1 and B2)** are shown as additional nodes connected to all neurons in the respective layers.

The visualization provides a clear understanding of how data flows through the network during forward propagation and how errors are propagated backward during training.
