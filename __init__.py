"""
Embedding layer - turns integer token ids into trainable dense vectors.
Normally the first layer in a text model.
"""

import numpy as np


class Layer_Embedding:
    """
    A trainable lookup table, shape (vocab_size, embedding_dim).

    This is normally the first layer in a text model - raw text gets
    turned into integer token ids *outside* this framework (e.g. a plain
    word -> index dictionary), and Layer_Embedding turns those ids into
    vectors a recurrent layer can do math on.

        forward:   inputs (N, T) int ids  ->  output (N, T, embedding_dim)
                                               (self.w[inputs], fancy indexing)
        backward:  d_output (N, T, embedding_dim)  ->  dw accumulated via
                                                        np.add.at(dw, inputs, d_output)

    Unlike Layer_Dense, vocab_size is known upfront, so the embedding
    table is initialized eagerly here in __init__ rather than lazily on
    the first forward() call. There's no d_inputs to propagate further
    back - token ids aren't differentiable, so this layer is only ever
    meant to go first.
    """

    param_names = ["w"]

    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.w = 0.01 * np.random.randn(vocab_size, embedding_dim)

    def forward(self, inputs):
        # inputs: (N, T) integer token ids
        self.inputs = inputs
        self.output = self.w[inputs]
        return self.output

    def backward(self, d_output):
        self.dw = np.zeros_like(self.w)

        # np.add.at matters here: a token appearing more than once in a
        # batch (very common - "the", "was", ...) needs its gradient
        # accumulated from every occurrence, not overwritten. Plain
        # self.dw[self.inputs] += d_output silently drops all but one
        # occurrence for repeated indices; np.add.at doesn't.
        np.add.at(self.dw, self.inputs, d_output)

        self.d_inputs = None
        return self.d_inputs
