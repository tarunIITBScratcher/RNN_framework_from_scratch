"""
Activation functions.

Every activation implements the same tiny interface:
    forward(inputs)  -> output          (and caches whatever backward() needs)
    backward(d_output) -> d_inputs      (gradient w.r.t. the layer's inputs)

This mirrors the interface used by Layer_Dense / Layer_Embedding /
Layer_SimpleRNN / Layer_LSTM / Layer_GRU, so a Model can chain layers and
activations interchangeably and just call .forward()/.backward() on
everything in the list.
"""

import numpy as np


class Activation_ReLU:
    """Rectified Linear Unit: f(x) = max(0, x)."""

    def forward(self, inputs):
        # inputs are cached because backward() needs to know which
        # positions were <= 0 (those get zero gradient).
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, d_output):
        self.d_inputs = d_output.copy()
        self.d_inputs[self.inputs <= 0] = 0
        return self.d_inputs


class Activation_LeakyReLU:
    """Leaky ReLU: f(x) = x if x > 0 else alpha * x."""

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(self.alpha * inputs, inputs)
        return self.output

    def backward(self, d_output):
        self.d_inputs = d_output.copy()
        self.d_inputs[self.inputs <= 0] *= self.alpha
        return self.d_inputs


class Activation_Sigmoid:
    """Sigmoid: f(x) = 1 / (1 + e^-x)."""

    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(-inputs)
        self.output = 1 / (1 + exp_values)
        return self.output

    def backward(self, d_output):
        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        self.d_inputs = d_output * self.output * (1 - self.output)
        return self.d_inputs


class Activation_Tanh:
    """
    Tanh: f(x) = tanh(x).

    New in this repo - it's the squashing function every recurrent cell
    below (SimpleRNN/LSTM/GRU) uses *internally*. Note that those cells
    don't actually call this class; the tanh math is inlined directly
    into their own forward()/backward(), because BPTT needs the cached
    hidden/cell states at *every* timestep, not just one layer's
    input/output. This class is provided as a standalone building block
    for stacking in a plain feed-forward Model if you ever want one.
    """

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, d_output):
        # d/dx tanh(x) = 1 - tanh(x)^2
        self.d_inputs = d_output * (1 - self.output ** 2)
        return self.d_inputs


class Activation_Softmax:
    """
    Softmax over the last axis, applied row-wise for a (samples, classes) input.

    Numerically stabilized by subtracting the row max before exponentiating.
    Used stand-alone (backward() below) or fused with categorical
    cross-entropy via Activation_Softmax_Loss_CategoricalCrossentropy for a
    much simpler/cheaper combined gradient.
    """

    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, d_output):
        # Full Jacobian-vector product for softmax, per sample:
        # dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
        self.d_inputs = self.output * (
            d_output - np.sum(d_output * self.output, axis=1, keepdims=True)
        )
        return self.d_inputs
