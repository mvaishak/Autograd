# Autgrad: Automatic Differentiation Engine

A complete neural network and automatic differentiation library built from scratch, demonstrating how automatic differentiation and neural networks work at a fundamental level.

## Overview

This project implements a fully functional autograd engine with support for:

- **Automatic Differentiation**: Backpropagation through computational graphs using the chain rule
- **Tensor Operations**: Addition, subtraction, multiplication, division, powers
- **Activation Functions**: ReLU and linear activations
- **Neural Network Layers**: Neurons, dense layers, and multi-layer perceptrons (MLPs)
- **Training**: Complete gradient descent optimization pipeline
- **Graph Visualization**: Computational graph visualization using Graphviz

## Project Structure

```
autograd_engine/
├── autgrad/                  # Main package
│   ├── __init__.py          # Package initialization
│   ├── tensor.py            # Core Tensor class with autodiff
│   ├── module.py            # Base Module class
│   ├── layers.py            # Neuron, Layer, MLP classes
│   └── utils.py             # Visualization utilities
├── tests/
│   └── test_engine.py       # Comprehensive unit tests
├── examples/
│   └── test_autgrad.ipynb   # Interactive notebook with examples
├── README.md                # This file
├── .gitignore               # Git ignore file
└── requirements.txt         # Python dependencies
```

## Installation

### Requirements
- Python 3.7+
- NumPy
- Graphviz (optional, for graph visualization)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd autograd_engine
```

2. Install the package:
```bash
pip install -e .
```

Or install with dev dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Tensor Operations

```python
from autgrad import Tensor

# Create tensors
a = Tensor(2.0)
b = Tensor(3.0)

# Perform operations
c = a * b
d = c + a

# Forward pass
print(d.data)  # 8.0 = (2 * 3) + 2

# Backward pass (compute gradients)
d.backward()
print(a.grad)  # 4.0
print(b.grad)  # 2.0
```

### Building a Neural Network

```python
from autgrad import MLP, Tensor

# Create a multi-layer perceptron
# 2 inputs -> 4 hidden (ReLU) -> 4 hidden (ReLU) -> 1 output (Linear)
model = MLP(nin=2, nouts=[4, 4, 1], 
            activations=['relu', 'relu', 'linear'])

# Forward pass
x = Tensor([1.0, 2.0])
output = model(x)
```

### Training a Model

```python
from autgrad import Tensor, MLP

# Create model and dataset
model = MLP(2, [4, 4, 1])
xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1]  # XOR problem

# Training loop
learning_rate = 0.01
for iteration in range(400):
    # Forward pass
    ypred = [model(Tensor(x)) for x in xs]
    ypred_tensor = Tensor.stack(ypred)
    ys_tensor = Tensor([[y] for y in ys])
    
    # Compute loss (MSE)
    diff = ypred_tensor - ys_tensor
    loss = (diff * diff).sum()
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Update parameters
    for p in model.parameters():
        p.data += -learning_rate * p.grad
    
    if iteration % 100 == 0:
        print(f"Loss: {loss.data:.4f}")
```

## Core Components

### Tensor Class

The `Tensor` class is the foundation of the entire system:

```python
class Tensor:
    def __init__(self, data, _children=()):
        self.data = np.array(data)              # Actual data
        self.grad = np.zeros_like(self.data)    # Gradient
        self._prev = set(_children)             # Parent nodes
        self._backward = lambda: None           # Backward function
```

**Supported Operations**:
- Arithmetic: `+`, `-`, `*`, `/`
- Power: `**` (including reverse power)
- Activation: `relu()`
- Utilities: `sum()`, `stack()`, `backward()`

### Module Class

Base class for all trainable components:

```python
class Module:
    def parameters(self):
        """Return list of trainable parameters."""
        return []
    
    def zero_grad(self):
        """Reset gradients to zero."""
        for param in self.parameters():
            param.grad = np.zeros_like(param.data)
```

### Network Layers

- **Neuron**: Single unit with weights and bias
  ```python
  neuron = Neuron(nin=3, activation='relu')
  output = neuron(x)
  ```

- **Layer**: Collection of neurons
  ```python
  layer = Layer(nin=3, nout=4, activation='relu')
  output = layer(x)  # Shape: (4,)
  ```

- **MLP**: Stack of layers
  ```python
  mlp = MLP(nin=3, nouts=[4, 4, 1], 
            activations=['relu', 'relu', 'linear'])
  output = mlp(x)
  ```

## Key Concepts

### Automatic Differentiation

The system uses the computational graph to track all operations. During backpropagation:

1. **Topological Sort**: Traverse the graph in reverse topological order
2. **Chain Rule**: Apply ∂L/∂a = ∂L/∂b · ∂b/∂a for each operation
3. **Gradient Accumulation**: Add gradients from multiple paths

### Broadcasting

Operations automatically handle tensors of different shapes:

```python
a = Tensor([[1, 2], [3, 4]])  # Shape: (2, 2)
b = Tensor([10, 20])           # Shape: (2,)
c = a + b                       # Broadcasts b to match a
```

Gradients are properly reduced when backpropagating through broadcasted operations.

### Activation Functions

- **ReLU**: max(0, x) - introduces non-linearity
- **Linear**: f(x) = x - for output layers

## Testing

Run the test suite:

```bash
# Using pytest
pytest tests/

# Using Python unittest
python -m pytest tests/test_engine.py -v
```

The test suite includes:
- Basic tensor operations
- Broadcasting behavior
- Gradient correctness (numerical vs analytical)
- Neural network layers
- Training functionality

## Examples

See `examples/test_autgrad.ipynb` for comprehensive examples including:

1. Basic tensor operations and gradients
2. Power and exponential operations
3. ReLU activation function
4. Single neurons and layers
5. Training on simple datasets
6. Solving the XOR problem
7. Computational graph visualization

To run the notebook:

```bash
jupyter notebook examples/test_autgrad.ipynb
```

## Mathematical Foundation

### Chain Rule

For a composite function:
$$\frac{dL}{da} = \frac{dL}{db} \cdot \frac{db}{da}$$

### Gradient Descent

Parameters are updated using:
$$w_{t+1} = w_t - \alpha \nabla L$$

where α is the learning rate and ∇L is the gradient of the loss.

### ReLU Activation

$$\text{ReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$$

Gradient:
$$\frac{d}{dx}\text{ReLU}(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$$


NOTE: 
This implementation prioritizes clarity and educational value over performance.

## Future Enhancements

- [ ] Matrix multiplication operator
- [ ] Softmax and cross-entropy loss
- [ ] More activation functions (sigmoid, tanh)
- [ ] Batch normalization
- [ ] Dropout
- [ ] Convolutional layers
- [ ] Recurrent layers
- [ ] GPU support


## License

MIT License - See LICENSE file for details

## Acknowledgments

This project was built to understand the fundamentals of automatic differentiation and neural networks. It draws inspiration from:

- PyTorch's design philosophy
- Deep Learning textbooks
- Educational implementations in similar projects

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Karpathy, A. (2017). *A Hacker's Guide to Artificial Intelligence*

---

**Last Updated**: January 2026
