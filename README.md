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

# Create input tensor
input_data = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
]

y_true = [
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5]
]

model = mixedNetwork(3,2,10)
optimizer = nn.Adam([[layer.parameters for layer in network] for network in model.networks], lr=Decimal(0.0005))
criterion = nn.L1Loss()

# Forward pass
while loss >= 0.00001:
     y_pred = model.forward(input_data) 

    # Calculate loss
    loss = criterion.forward(y_pred,batch_labels)
    
    print('Loss: ' + str(loss) )
    loss_grads = criterion.backward(y_pred,y_true)                
    gradients = model.backward(loss_grads)

    # Update parameters with Adam optimizer
    optimizer.step(gradients)
    model.update_parameters(optimizer.parameters)
    
    with open('Models/customNetModel.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('Models/optimizer_state.pkl', 'wb') as f:
        pickle.dump(optimizer, f)

```

## Contributing
Please feel free to submit issues and pull requests for any bug fixes, improvements, or new features.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
