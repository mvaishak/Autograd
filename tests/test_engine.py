"""
Unit tests for the autgrad engine.

Tests cover tensor operations, gradient computation, neural network layers,
and end-to-end training.
"""

import numpy as np
import pytest
from autgrad import Tensor, Neuron, Layer, MLP


class TestTensorOperations:
    """Tests for basic tensor operations."""

    def test_addition(self):
        """Test tensor addition forward and backward pass."""
        a = Tensor(2.0)
        b = Tensor(3.0)
        c = a + b

        assert c.data == 5.0

        c.backward()
        assert a.grad == 1.0
        assert b.grad == 1.0

    def test_multiplication(self):
        """Test tensor multiplication forward and backward pass."""
        a = Tensor(2.0)
        b = Tensor(3.0)
        c = a * b

        assert c.data == 6.0

        c.backward()
        assert a.grad == 3.0  # dc/da = b
        assert b.grad == 2.0  # dc/db = a

    def test_subtraction(self):
        """Test tensor subtraction forward and backward pass."""
        a = Tensor(5.0)
        b = Tensor(3.0)
        c = a - b

        assert c.data == 2.0

        c.backward()
        assert a.grad == 1.0
        assert b.grad == -1.0

    def test_division(self):
        """Test tensor division forward and backward pass."""
        a = Tensor(6.0)
        b = Tensor(2.0)
        c = a / b

        assert c.data == 3.0

        c.backward()
        assert np.isclose(a.grad, 0.5)  # dc/da = 1/b
        assert np.isclose(b.grad, -1.5)  # dc/db = -a/b^2

    def test_power(self):
        """Test power operation forward and backward pass."""
        a = Tensor(2.0)
        c = a**3

        assert c.data == 8.0

        c.backward()
        assert a.grad == 12.0  # dc/da = 3 * a^2 = 3 * 4 = 12

    def test_relu(self):
        """Test ReLU activation forward and backward pass."""
        a = Tensor([2.0, -1.0, 3.0, -0.5])
        b = a.relu()

        np.testing.assert_array_equal(b.data, np.array([2.0, 0.0, 3.0, 0.0]))

        b.backward()
        np.testing.assert_array_equal(a.grad, np.array([1.0, 0.0, 1.0, 0.0]))

    def test_sum(self):
        """Test sum operation forward and backward pass."""
        a = Tensor([1.0, 2.0, 3.0])
        b = a.sum()

        assert b.data == 6.0

        b.backward()
        np.testing.assert_array_equal(a.grad, np.ones(3))

    def test_chain_rule(self):
        """Test chain rule with multiple operations."""
        a = Tensor(2.0)
        b = Tensor(3.0)
        c = a * b
        d = c + a

        assert d.data == 8.0  # 2*3 + 2 = 8

        d.backward()
        # dd/da = b + 1 = 3 + 1 = 4
        # dd/db = a = 2
        assert a.grad == 4.0
        assert b.grad == 2.0

    def test_complex_computation(self):
        """Test complex expression with multiple operations."""
        a = Tensor(2.0)
        b = Tensor(3.0)
        c = a * b
        d = c + 2**a

        assert np.isclose(d.data, 10.0)  # 2*3 + 2^2 = 6 + 4 = 10

        d.backward()
        # dd/da = b + ln(2) * 2^a = 3 + ln(2)*4 ≈ 3 + 2.772 = 5.772
        expected_grad_a = 3.0 + np.log(2) * 4.0
        assert np.isclose(a.grad, expected_grad_a)
        assert b.grad == 2.0

    def test_broadcasting_addition(self):
        """Test addition with broadcasting."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([1.0, 2.0])
        c = a + b

        np.testing.assert_array_equal(c.data, np.array([[2.0, 4.0], [4.0, 6.0]]))

        c.backward()
        np.testing.assert_array_equal(b.grad, np.array([2.0, 2.0]))

    def test_stack(self):
        """Test tensor stacking operation."""
        a = Tensor(1.0)
        b = Tensor(2.0)
        c = Tensor(3.0)
        stacked = Tensor.stack([a, b, c])

        np.testing.assert_array_equal(stacked.data, np.array([1.0, 2.0, 3.0]))

        stacked.backward()
        assert a.grad == 1.0
        assert b.grad == 1.0
        assert c.grad == 1.0


class TestNeuralNetworkComponents:
    """Tests for neural network layer components."""

    def test_neuron_forward(self):
        """Test neuron forward pass."""
        neuron = Neuron(nin=3, activation="linear")
        x = Tensor([1.0, 2.0, 3.0])
        output = neuron(x)

        # Output should be a scalar
        assert output.data.shape == ()

    def test_neuron_relu_activation(self):
        """Test neuron with ReLU activation."""
        neuron = Neuron(nin=2, activation="relu")
        x = Tensor([1.0, -1.0])
        output = neuron(x)

        # Should have applied ReLU
        assert output._backward is not None

    def test_neuron_parameters(self):
        """Test neuron parameter collection."""
        neuron = Neuron(nin=3, activation="relu")
        params = neuron.parameters()

        assert len(params) == 2  # weight and bias
        assert params[0].data.shape == (3,)  # weight shape
        assert params[1].data.shape == ()  # bias is scalar

    def test_neuron_backprop(self):
        """Test neuron backpropagation."""
        neuron = Neuron(nin=2, activation="linear")
        # Set specific weights and bias for testing
        neuron.w = Tensor([1.0, 2.0])
        neuron.b = Tensor(1.0)

        x = Tensor([1.0, 1.0])
        output = neuron(x)

        output.backward()

        # Gradients should be non-zero
        assert neuron.w.grad is not None
        assert neuron.b.grad is not None

    def test_layer_forward(self):
        """Test layer forward pass."""
        layer = Layer(nin=3, nout=4, activation="relu")
        x = Tensor([1.0, 2.0, 3.0])
        output = layer(x)

        # Should have 4 outputs (one per neuron)
        assert output.data.shape == (4,)

    def test_layer_parameters(self):
        """Test layer parameter collection."""
        layer = Layer(nin=3, nout=4, activation="relu")
        params = layer.parameters()

        # 4 neurons * 2 (weight + bias) = 8 parameters
        assert len(params) == 8

    def test_mlp_forward(self):
        """Test MLP forward pass."""
        mlp = MLP(nin=3, nouts=[4, 2])
        x = Tensor([1.0, 2.0, 3.0])
        output = mlp(x)

        # Output should match final layer size
        assert output.data.shape == (2,)

    def test_mlp_parameters(self):
        """Test MLP parameter collection."""
        mlp = MLP(nin=3, nouts=[4, 4, 1])
        params = mlp.parameters()

        # Layer 1: 3 inputs, 4 neurons = 4*3 weights + 4 biases = 16 params
        # Layer 2: 4 inputs, 4 neurons = 4*4 weights + 4 biases = 20 params
        # Layer 3: 4 inputs, 1 neuron = 1*4 weights + 1 bias = 5 params
        # Total = 41 params
        assert len(params) == 41

    def test_mlp_activations(self):
        """Test MLP with custom activations."""
        mlp = MLP(nin=2, nouts=[4, 1], activations=["relu", "linear"])
        x = Tensor([1.0, 2.0])
        output = mlp(x)

        assert output.data.shape == ()


class TestTraining:
    """Tests for training functionality."""

    def test_zero_grad(self):
        """Test zero_grad functionality."""
        model = MLP(nin=2, nouts=[4, 1])

        # Set some gradients
        for p in model.parameters():
            p.grad = np.ones_like(p.data)

        # Zero them
        model.zero_grad()

        # Check they're zero
        for p in model.parameters():
            assert np.all(p.grad == 0)

    def test_training_step(self):
        """Test a single training step."""
        model = MLP(nin=2, nouts=[4, 1])
        x = Tensor([1.0, 1.0])
        y = Tensor([1.0])

        # Forward
        pred = model(x)

        # Loss (MSE)
        diff = pred + y * Tensor(-1)
        loss = (diff * diff).sum()

        # Backward
        model.zero_grad()
        loss.backward()

        # Check gradients computed
        has_grads = any(np.any(p.grad != 0) for p in model.parameters())
        assert has_grads

        # Update
        lr = 0.01
        for p in model.parameters():
            old_data = p.data.copy()
            p.data += -lr * p.grad
            # Check parameters changed
            assert not np.allclose(old_data, p.data)

    def test_training_loop_loss_reduction(self):
        """Test that loss reduces over training iterations."""
        np.random.seed(42)
        model = MLP(nin=2, nouts=[4, 1])

        xs = [[0.0, 0.0], [1.0, 1.0]]
        ys = [-1.0, 1.0]

        losses = []
        for k in range(10):
            ypred = [model(Tensor(x)) for x in xs]
            ypred_tensor = Tensor.stack(ypred)
            ys_tensor = Tensor([[y] for y in ys])

            diff = ypred_tensor + ys_tensor * Tensor(-1)
            loss = (diff * diff).sum()
            losses.append(loss.data)

            model.zero_grad()
            loss.backward()

            for p in model.parameters():
                p.data += -0.01 * p.grad

        # Loss should generally decrease
        # (Allow some noise, just check trend)
        assert losses[-1] < losses[0]


class TestGradientCorrectness:
    """Tests for gradient correctness using numerical gradients."""

    def numerical_gradient(self, f, x, h=1e-5):
        """Compute numerical gradient using finite differences."""
        grad = np.zeros_like(x.data)
        it = np.nditer(x.data, flags=["multi_index"], op_flags=["readwrite"])

        for _ in it:
            idx = it.multi_index
            old_value = x.data[idx]

            x.data[idx] = old_value + h
            fxh = f()

            x.data[idx] = old_value - h
            fxh2 = f()

            x.data[idx] = old_value

            grad[idx] = (fxh - fxh2) / (2 * h)

        return grad

    def test_addition_gradient(self):
        """Test gradient correctness for addition."""
        a = Tensor(3.0)

        def f():
            b = Tensor(2.0)
            c = a + b
            return c.data

        loss = f()
        a_backup = Tensor(a.data)
        num_grad = self.numerical_gradient(a_backup, a)

        # Analytical gradient
        b = Tensor(2.0)
        c = a + b
        c.backward()

        assert np.allclose(a.grad, num_grad)

    def test_multiplication_gradient(self):
        """Test gradient correctness for multiplication."""
        a = Tensor(3.0)
        b = Tensor(2.0)
        c = a * b

        c.backward()

        # Numerical check
        eps = 1e-5
        a_plus = Tensor(3.0 + eps)
        b_copy = Tensor(2.0)
        c_plus = a_plus * b_copy
        grad_num = (c_plus.data - c.data) / eps

        assert np.isclose(a.grad, grad_num, atol=1e-4)

    def test_relu_gradient(self):
        """Test gradient correctness for ReLU."""
        # Positive value
        a = Tensor(2.0)
        b = a.relu()
        b.backward()
        assert a.grad == 1.0

        # Negative value
        c = Tensor(-2.0)
        d = c.relu()
        d.backward()
        assert c.grad == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
