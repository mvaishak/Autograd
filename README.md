# Autgrad: A Custom Autograd Engine

**Autgrad** is a lightweight deep learning library built from scratch.

I started this project because I wanted to look under the hood of frameworks like PyTorch. While it's easy to import a library and train a model, I wanted to understand exactly how the machine learns—specifically, how gradients flow backward through a dynamic computational graph.

This is a personal initiative to bridge the gap between mathematical theory and working software.

## Project Overview

In its current state, Autgrad focuses on the fundamentals: a scalar-value Autograd engine and the essential components to build a **Multilayer Perceptron (MLP)**. It builds a graph of operations and calculates gradients automatically, allowing for the training of simple neural networks.

## Directory Structure

```text
autograd_engine/
├── autgrad/
│   ├── tensor.py      # The core engine: handles data, gradients, and backprop
│   ├── module.py      # Base class for all neural network modules
│   ├── layers.py      # Implementations of Neurons, Layers, and MLPs
│   └── utils.py       # Tools for visualizing the computational graph
├── examples/
│   └── test_autgrad.ipynb  # Notebooks demonstrating training (e.g., XOR problem)
└── requirements.txt   # Dependencies (NumPy and Graphviz)

```

## Key Components

### 1. The Engine (`tensor.py`)

This is the heart of the library. The `Tensor` object wraps raw data (NumPy arrays) but adds a crucial feature: **memory**.

* It remembers what operation created it (addition, multiplication, power, etc.).
* It remembers its "parents" (the tensors that went into that operation).
* **Backpropagation:** When you call `.backward()`, the tensor recursively calculates gradients for all its ancestors using the Chain Rule.

### 2. Neural Network Modules (`layers.py`)

Built on top of the engine, these are the building blocks for models:

* **Neuron:** A single unit with weights and a bias.
* **Layer:** A collection of neurons operating in parallel.
* **MLP:** A stack of layers with non-linear activation functions (ReLU).

## Installation

You need Python 3.7+ and a few dependencies.

```bash
# Clone the repo
git clone <your-repo-url>
cd autograd_engine

# Install dependencies
pip install -r requirements.txt

```

## Usage Example: Solving XOR

You can build and train a model just like you would in PyTorch. Here is how you train a simple MLP to solve the XOR problem:

```python
from autgrad import Tensor, MLP

# 1. Define the model
# 2 inputs -> 4 hidden units -> 4 hidden units -> 1 output
model = MLP(nin=2, nouts=[4, 4, 1], activations=['relu', 'relu', 'linear'])

# 2. Define the dataset (XOR)
xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1] 

# 3. Training Loop
for k in range(400):
    
    # Forward pass
    ypred = [model(Tensor(x)) for x in xs]
    
    # Loss (Mean Squared Error)
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    # Backward pass
    model.zero_grad()  # Reset gradients
    loss.backward()    # Backpropagate through the graph
    
    # Update weights (Stochastic Gradient Descent)
    for p in model.parameters():
        p.data += -0.01 * p.grad
        
    if k % 100 == 0:
        print(f"Step {k} | Loss: {loss.data:.4f}")

```

## Roadmap

This project is part of my ongoing study of deep learning internals. Future plans include:

* [ ] **Matrix Multiplication:** optimizing the engine to handle efficient matrix ops rather than scalar loops.
* [ ] **ConvNets:** Implementing Convolutional layers for image processing.
* [ ] **Optimizers:** Adding Adam and RMSProp.
* [ ] **CrossEntropy:** Implementing Softmax and proper classification loss.

## References & Inspiration

* **Andrej Karpathy:** His "Micrograd" lecture was a massive inspiration for the structure of this engine.
* **Deep Learning (Goodfellow et al.):** For the mathematical foundations.

---

*Note: This is an educational project not intended for production use. It favors code readability and simplicity over raw performance.*
