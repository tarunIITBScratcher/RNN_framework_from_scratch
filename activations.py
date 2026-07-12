"""
rnn_framework: a from-scratch (NumPy only) recurrent neural network
framework - Embedding, SimpleRNN, LSTM, GRU (all many-to-one), plus the
same activations/losses/optimizers/Dense/Model machinery as
ann_framework/cnn_framework, so this package is fully self-contained and
installable on its own. Built around one concrete target task: sentiment
analysis.
"""

from .activations import (
    Activation_ReLU,
    Activation_LeakyReLU,
    Activation_Sigmoid,
    Activation_Tanh,
    Activation_Softmax,
)
from .losses import (
    Loss_CategoricalCrossentropy,
    Activation_Softmax_Loss_CategoricalCrossentropy,
    Squared_Loss,
    Loss_BinaryCrossentropy,
    Activation_Sigmoid_Loss_BinaryCrossentropy,
)
from .optimizers import (
    Optimizer_normalGD,
    Optimizer_GD_decay,
    Optimizer_momentum,
    Optimizer_Adagrad,
    Optimizer_RMSprop,
    Optimizer_ADAM,
)
from .layers_dense import Layer_Dense
from .layers_embedding import Layer_Embedding
from .layers_recurrent import Layer_SimpleRNN, Layer_LSTM, Layer_GRU
from .model import Model

__all__ = [
    "Activation_ReLU",
    "Activation_LeakyReLU",
    "Activation_Sigmoid",
    "Activation_Tanh",
    "Activation_Softmax",
    "Loss_CategoricalCrossentropy",
    "Activation_Softmax_Loss_CategoricalCrossentropy",
    "Squared_Loss",
    "Loss_BinaryCrossentropy",
    "Activation_Sigmoid_Loss_BinaryCrossentropy",
    "Optimizer_normalGD",
    "Optimizer_GD_decay",
    "Optimizer_momentum",
    "Optimizer_Adagrad",
    "Optimizer_RMSprop",
    "Optimizer_ADAM",
    "Layer_Dense",
    "Layer_Embedding",
    "Layer_SimpleRNN",
    "Layer_LSTM",
    "Layer_GRU",
    "Model",
]

__version__ = "1.0.0"
