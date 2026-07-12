"""
Fully-connected layer - unchanged from ann_framework/cnn_framework, plus
a `param_names` declaration for the generalized optimizer contract (see
optimizers.py).
"""

import numpy as np


class Layer_Dense:
    """
    Fully-connected layer with lazy weight initialization.

    You only pass n_neurons when building the model - the input dimension
    is inferred from the first forward() call, so you never need to
    hardcode `input_shape` when stacking layers.
    """

    param_names = ["w", "b"]

    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.initialized = False

    def forward(self, inputs):
        if not self.initialized:
            # -----
            input_dim = inputs.shape[1]
            self.w = 0.01 * np.random.randn(input_dim, self.n_neurons)
            self.b = np.zeros((1, self.n_neurons))
            self.initialized = True
            # -----
            # lazy initialization so we don't need to define inputs when
            # defining layers in the model

        self.inputs = inputs
        self.output = np.dot(inputs, self.w) + self.b

        return self.output

    def backward(self, d_output):
        self.dw = np.dot(self.inputs.T, d_output)
        self.db = np.sum(d_output, axis=0, keepdims=True)
        self.d_inputs = np.dot(d_output, self.w.T)
        return self.d_inputs
