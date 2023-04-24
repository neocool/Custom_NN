# Custom Decimal Neural Network

This repository contains a custom implementation of a neural network with Decimal arithmetic, making use of Python's `decimal` library. The primary objective of this implementation is to provide higher precision when working with floating-point numbers in deep learning models.

## Features

- Custom neural network architecture with Decimal arithmetic
- Custom layers, activation functions, loss functions, and optimizers with Decimal support
- Utility functions for Decimal arithmetic operations on lists and tensors

## Getting Started

### Requirements

- Python 3.7+
- `decimal` library (built-in)

### Usage

To use the Custom Decimal Neural Network in your project, simply import the necessary classes and utility functions from the provided Python files.

Here's a sample usage:

```python
import NN_Module

# Create a neural network with a Dense layer and Sigmoid activation
nn = NeuralNetwork([DenseLayer(3, 2), Sigmoid()])

# Create input tensor
input_data = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
]
input_tensor = nntensor(input_data)

# Forward pass
output = nn.forward(input_tensor)

# Calculate loss
y_true = [
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5]
]
loss_function = L1Loss()
loss = loss_function.forward(output, y_true)

# Backward pass
gradients = loss_function.backward(output, y_true)
nn.backward(gradients)

# Update parameters with Adam optimizer
optimizer = Adam(nn.parameters, lr=0.001)
optimizer.step(gradients)

```

## Contributing
Please feel free to submit issues and pull requests for any bug fixes, improvements, or new features.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
