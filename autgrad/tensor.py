"""
Tensor class for automatic differentiation (autograd).

This module implements a Tensor class that tracks computational graphs
and enables automatic differentiation through backpropagation.
"""

import numpy as np


class Tensor:
    """A Tensor class with automatic differentiation support.
    
    Stores data, gradients, and maintains a computational graph for
    backpropagation. Supports various operations with automatic gradient
    computation.
    
    Attributes:
        data (np.ndarray): The tensor data stored as a numpy array.
        grad (np.ndarray): Gradient with respect to loss (initialized to zeros).
        _prev (set): Set of child tensors in the computational graph.
        _backward (callable): Function to execute during backpropagation.
    """
    
    def __init__(self, data, _children=()):
        """Initialize a Tensor.
        
        Args:
            data: Input data (list, number, or numpy array).
            _children: Tuple of child tensors in the computational graph.
        """
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._prev = set(_children)
        self._backward = lambda: None

    def sum(self):
        """Sum all elements in the tensor.
        
        Returns:
            Tensor: A scalar tensor containing the sum.
        """
        out = Tensor(np.sum(self.data), _children=(self,))
        
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        
        out._backward = _backward
        return out

    def __add__(self, other):
        """Element-wise addition.
        
        Args:
            other: Tensor or scalar to add.
            
        Returns:
            Tensor: Result of addition.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other))

        def _backward():
            grad_self = out.grad
            grad_other = out.grad
            # Sum out added dims (broadcasting)
            ndims_self = len(self.data.shape)
            ndims_out = len(out.grad.shape)
            for _ in range(ndims_out - ndims_self):
                grad_self = grad_self.sum(axis=0)
            ndims_other = len(other.data.shape)
            for _ in range(ndims_out - ndims_other):
                grad_other = grad_other.sum(axis=0)
            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    def __mul__(self, other):
        """Element-wise multiplication.
        
        Args:
            other: Tensor or scalar to multiply.
            
        Returns:
            Tensor: Result of multiplication.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other))

        def _backward():
            grad_self = other.data * out.grad
            grad_other = self.data * out.grad
            # Handle broadcasting
            ndims_self = len(self.data.shape)
            ndims_out = len(out.grad.shape)
            for _ in range(ndims_out - ndims_self):
                grad_self = grad_self.sum(axis=0)
            ndims_other = len(other.data.shape)
            for _ in range(ndims_out - ndims_other):
                grad_other = grad_other.sum(axis=0)
            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    def __sub__(self, other):
        """Element-wise subtraction.
        
        Args:
            other: Tensor or scalar to subtract.
            
        Returns:
            Tensor: Result of subtraction.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, _children=(self, other))

        def _backward():
            grad_self = out.grad
            grad_other = -out.grad
            # Handle broadcasting
            ndims_self = len(self.data.shape)
            ndims_out = len(out.grad.shape)
            for _ in range(ndims_out - ndims_self):
                grad_self = grad_self.sum(axis=0)
            ndims_other = len(other.data.shape)
            for _ in range(ndims_out - ndims_other):
                grad_other = grad_other.sum(axis=0)
            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    def __truediv__(self, other):
        """Element-wise division.
        
        Args:
            other: Tensor or scalar to divide by.
            
        Returns:
            Tensor: Result of division.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, _children=(self, other))

        def _backward():
            grad_self = (1 / other.data) * out.grad
            grad_other = -(self.data / (other.data ** 2)) * out.grad
            # Handle broadcasting
            ndims_self = len(self.data.shape)
            ndims_out = len(out.grad.shape)
            for _ in range(ndims_out - ndims_self):
                grad_self = grad_self.sum(axis=0)
            ndims_other = len(other.data.shape)
            for _ in range(ndims_out - ndims_other):
                grad_other = grad_other.sum(axis=0)
            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    def __neg__(self):
        """Negation (unary minus).
        
        Returns:
            Tensor: Negated tensor.
        """
        out = Tensor(-self.data, _children=(self,))

        def _backward():
            self.grad -= out.grad

        out._backward = _backward
        return out

    def __pos__(self):
        """Unary plus (identity).
        
        Returns:
            Tensor: Copy of tensor.
        """
        out = Tensor(+self.data, _children=(self,))

        def _backward():
            self.grad += out.grad

        out._backward = _backward
        return out

    def __pow__(self, power):
        """Power operation (self ** power).
        
        Args:
            power: Exponent (scalar).
            
        Returns:
            Tensor: Result of power operation.
        """
        out = Tensor(self.data ** power, _children=(self,))

        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __rpow__(self, base):
        """Reverse power operation (base ** self).
        
        Args:
            base: Base (scalar).
            
        Returns:
            Tensor: Result of reverse power operation.
        """
        out = Tensor(base ** self.data, _children=(self,))

        def _backward():
            self.grad += (np.log(base) * base ** self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        """Rectified Linear Unit activation.
        
        Returns:
            Tensor: Result after applying ReLU activation.
        """
        out = Tensor(np.maximum(0, self.data), _children=(self,))

        def _backward():
            self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def backward(self):
        """Perform backpropagation through the computational graph.
        
        Computes gradients for all tensors in the graph by traversing
        in reverse topological order.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.data, dtype=np.float64)

        for node in reversed(topo):
            node._backward()

    @staticmethod
    def stack(tensors, axis=0):
        """Stack multiple tensors along a new axis.
        
        Args:
            tensors: List of Tensors to stack.
            axis: Axis along which to stack (default: 0).
            
        Returns:
            Tensor: Stacked tensor with gradients properly backpropagated.
        """
        data = np.stack([t.data for t in tensors], axis=axis)
        out = Tensor(data, _children=tuple(tensors))

        def _backward():
            for i, t in enumerate(tensors):
                t.grad += np.take(out.grad, indices=i, axis=axis)

        out._backward = _backward
        return out

    def __repr__(self):
        """String representation of tensor."""
        return f"Tensor(data={self.data}, grad={self.grad})"
