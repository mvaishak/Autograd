"""
Base Module class for neural network components.

Provides a base class for all neural network modules that need to
track and manage parameters and their gradients.
"""

import numpy as np


class Module:
    """Base class for neural network modules.
    
    Provides functionality to track parameters and manage gradient
    updates during training.
    """

    def parameters(self):
        """Return list of all parameters in this module.
        
        Should be overridden by subclasses to return their parameters.
        
        Returns:
            list: List of Tensor objects that are parameters.
        """
        return []

    def zero_grad(self):
        """Reset gradients of all parameters to zero.
        
        Should be called before each backward pass to avoid accumulating
        gradients across multiple training steps.
        """
        for param in self.parameters():
            param.grad = np.zeros_like(param.data)
