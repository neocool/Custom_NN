# Decimal Nerual Network
This repository contains a custom implementation of the Transformer model with Decimal arithmetic, making use of Python's decimal library. The primary objective of this implementation is to provide higher precision when working with floating-point numbers in deep learning models, specifically the Transformer architecture.

Features
Customized Transformer architecture with Decimal arithmetic
Custom Layer Normalization, Multi-head Attention, and Position-wise Feed-forward Network layers
Custom L1 Loss and Adam optimizer with Decimal support
Utility functions for Decimal arithmetic operations on lists and tensors
Getting Started
Requirements
Python 3.7+
decimal library (built-in)
Usage
To use the Decimal Transformer in your project, simply import the necessary classes and utility functions from the provided Python files.

Here's a sample usage:

python
Copy code
from decimal_transformer import TransformerLayer
from decimal_utils import nntensor

# Create a Transformer layer
transformer_layer = TransformerLayer(d_model=512, num_heads=8, d_ff=2048)

# Create input tensor
input_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
input_tensor = nntensor(input_data)

# Forward pass
output = transformer_layer.forward(input_tensor)
Contributing
Please feel free to submit issues and pull requests for any bug fixes, improvements, or new features.

License
This project is licensed under the MIT License. See the LICENSE file for more details.
