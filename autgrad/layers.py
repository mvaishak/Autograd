"""
Neural network layers and components.

Implements Neuron, Layer, and MLP classes for building neural networks.
"""

import numpy as np
from .tensor import Tensor
from .module import Module


class Neuron(Module):
    """A single neuron that performs a linear transformation followed by activation.
    
    The neuron computes: activation(x · w + b) where w and b are learned parameters.
    
    Attributes:
        w (Tensor): Weight vector for inputs.
        b (Tensor): Bias term.
        activation (str): Activation function name ('relu' or 'linear').
    """

    def __init__(self, nin, activation="relu"):
        """Initialize a neuron.
        
        Args:
            nin (int): Number of input features.
            activation (str): Activation function ('relu' or 'linear'). Default: 'relu'.
        """
        self.w = Tensor(np.random.uniform(-1, 1, size=nin))
        self.b = Tensor(np.random.uniform(-1, 1))
        self.activation = activation

    def parameters(self):
        """Return weight and bias parameters.
        
        Returns:
            list: [weight, bias] Tensors.
        """
        return [self.w, self.b]

    def __call__(self, x):
        """Forward pass through the neuron.
        
        Args:
            x: Input (Tensor or array-like).
            
        Returns:
            Tensor: Output after linear transformation and activation.
        """
        xi = x if isinstance(x, Tensor) else Tensor(x)
        act = (xi * self.w).sum() + self.b
        return act.relu() if self.activation == "relu" else act


class Layer(Module):
    """A layer consisting of multiple neurons.
    
    All neurons in a layer share the same input but apply different weights
    and biases, producing multiple outputs.
    
    Attributes:
        neurons (list): List of Neuron objects.
    """

    def __init__(self, nin, nout, activation="relu"):
        """Initialize a layer.
        
        Args:
            nin (int): Number of input features per neuron.
            nout (int): Number of neurons (outputs) in the layer.
            activation (str): Activation function for all neurons. Default: 'relu'.
        """
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        """Forward pass through all neurons in the layer.
        
        Args:
            x: Input to all neurons.
            
        Returns:
            Tensor: Stacked outputs from all neurons, shape (nout,).
        """
        out = [neuron(x) for neuron in self.neurons]
        return Tensor.stack(out, axis=0)

    def parameters(self):
        """Return all parameters from all neurons.
        
        Returns:
            list: All weight and bias Tensors from all neurons.
        """
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class MLP(Module):
    """Multi-Layer Perceptron (fully connected neural network).
    
    Stacks multiple layers together to create a deep neural network.
    
    Attributes:
        layers (list): List of Layer objects.
        activations (list): Activation functions for each layer.
    """

    def __init__(self, nin, nouts, activations=None):
        """Initialize an MLP.
        
        Args:
            nin (int): Number of input features.
            nouts (list): Number of neurons in each layer.
                Example: [64, 32, 10] creates 3 layers with 64, 32, 10 neurons.
            activations (list, optional): Activation function for each layer.
                If None, defaults to 'relu' for all hidden layers and 'linear' for output.
                Example: ['relu', 'relu', 'linear']
                
        Raises:
            ValueError: If number of activations doesn't match number of layers.
        """
        self.layers = []
        self.nin = nin

        if activations is None:
            activations = ["relu"] * (len(nouts) - 1) + ["linear"]

        if len(activations) != len(nouts):
            raise ValueError(
                "Number of activations must match number of layers (nouts)"
            )

        self.activations = activations

        for i, nout in enumerate(nouts):
            self.layers.append(Layer(self.nin, nout, activations[i]))
            self.nin = nout

    def __call__(self, x):
        """Forward pass through all layers.
        
        Args:
            x: Input data.
            
        Returns:
            Tensor: Output from the final layer.
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self):
        """Return all parameters from all layers.
        
        Returns:
            list: All weight and bias Tensors from all layers.
        """
        return [p for layer in self.layers for p in layer.parameters()]
