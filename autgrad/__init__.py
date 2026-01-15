"""
Autgrad: Automatic differentiation engine from scratch.

A simple but complete neural network and autograd library built from first principles,
demonstrating how automatic differentiation and neural networks work internally.

Main components:
- Tensor: Core data structure with automatic differentiation support
- Module: Base class for neural network components
- Layers: Neuron, Layer, and MLP classes for building networks
- Utils: Visualization tools for computational graphs
"""

from .tensor import Tensor
from .module import Module
from .layers import Neuron, Layer, MLP
from .utils import trace, draw_dot

__all__ = [
    "Tensor",
    "Module",
    "Neuron",
    "Layer",
    "MLP",
    "trace",
    "draw_dot",
]

__version__ = "1.0.0"
