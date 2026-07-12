"""
The Model class: a simple sequential container that wires layers,
activations, a loss, and an optimizer together into a trainable model.
Same training loop as ann_framework/cnn_framework's Model, generalized in
two ways for recurrent layers - see "How this differs" in the README:

  1. compile() also fuses Activation_Sigmoid + Loss_BinaryCrossentropy
     (in addition to the existing Softmax + CategoricalCrossentropy
     fusion), since sentiment analysis is a two-class problem modelled
     with one sigmoid output neuron.
  2. fit() finds trainable layers via hasattr(layer, "param_names")
     instead of isinstance() - so Layer_Embedding/Layer_SimpleRNN/
     Layer_LSTM/Layer_GRU are automatically picked up as trainable with
     no changes needed here.
"""

import numpy as np
import matplotlib.pyplot as plt

from .activations import Activation_Softmax, Activation_Sigmoid
from .losses import (
    Activation_Softmax_Loss_CategoricalCrossentropy,
    Loss_CategoricalCrossentropy,
    Activation_Sigmoid_Loss_BinaryCrossentropy,
    Loss_BinaryCrossentropy,
)


class Model:
    """
    Sequential model: holds an ordered list of layers/activations and
    drives the forward pass, backward pass, parameter updates, and a
    full train/validation training loop.
    """

    def __init__(self, layers):
        self.layers = layers
        self.loss = None
        self.optimizer = None

    def compile(self, optimizer, loss):
        """
        Attach an optimizer and a loss function to the model.

        If the last layer is Activation_Softmax and the chosen loss is
        Loss_CategoricalCrossentropy, they are automatically fused into
        Activation_Softmax_Loss_CategoricalCrossentropy. Likewise, if the
        last layer is Activation_Sigmoid and the chosen loss is
        Loss_BinaryCrossentropy (the sentiment-analysis case), they are
        fused into Activation_Sigmoid_Loss_BinaryCrossentropy. Both give
        a simpler, more numerically stable combined backward pass. In
        either case the standalone activation layer is popped off
        self.layers since the fused loss now performs that activation
        itself.
        """
        self.optimizer = optimizer

        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(
            loss, Loss_CategoricalCrossentropy
        ):
            self.loss = Activation_Softmax_Loss_CategoricalCrossentropy()
            self.layers.pop()
        elif isinstance(self.layers[-1], Activation_Sigmoid) and isinstance(
            loss, Loss_BinaryCrossentropy
        ):
            self.loss = Activation_Sigmoid_Loss_BinaryCrossentropy()
            self.layers.pop()
        else:
            self.loss = loss

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, dloss):
        grad = dloss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def _predicted_and_true_labels(self, probs, y_batch):
        """
        Turn raw probabilities + targets into comparable label arrays for
        accuracy, branching on the output shape: argmax for a softmax
        (N, classes) output, threshold-at-0.5 for a sigmoid (N, 1)
        output. `probs` here is the fused loss's cached self.loss.y_pred
        (actual probabilities), not the raw logits that self.forward()
        returns after compile() has popped the standalone activation -
        see the "watch out at inference time" note in losses.py.
        """
        if probs.shape[-1] == 1:
            pred_labels = (probs > 0.5).astype(int).flatten()
            true_labels = np.asarray(y_batch).flatten()
        else:
            pred_labels = np.argmax(probs, axis=1)
            true_labels = (
                np.argmax(y_batch, axis=1) if len(np.shape(y_batch)) == 2 else y_batch
            )
        return pred_labels, true_labels

    def fit(self, X, y, validation_split=0.2, epochs=1, batch_size=None, verbose=1):
        """
        Train the model with mini-batch gradient descent.

        Although each mini-batch contains different data, the model is
        still optimizing the same overall loss function on the full
        dataset. A mini-batch only provides a smaller sample of the data,
        so its gradient acts as a noisy estimate of the true gradient of
        the complete loss. Since the weights are shared and continuously
        updated across batches (not reset each time), the optimizer
        gradually moves toward minimizing the same objective, just using
        cheaper approximate gradient steps instead of computing the exact
        gradient on the entire dataset every update.
        """
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        X_train = X
        y_train = y
        X_val = None
        y_val = None

        if validation_split is not None:
            split_index = int(len(X) * (1 - validation_split))
            X_train = X[:split_index]
            y_train = y[:split_index]
            X_val = X[split_index:]
            y_val = y[split_index:]

        if batch_size is None:
            batch_size = len(X_train)

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        for epoch in range(epochs):
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            epoch_acc = 0
            batches = 0

            for start in range(0, len(X_train), batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self.forward(X_batch)
                loss = self.loss.forward(y_pred, y_batch)
                epoch_loss += loss
                batches += 1

                self.loss.backward()
                self.backward(self.loss.d_input)

                self.optimizer.pre_update_params()
                for layer in self.layers:
                    if hasattr(layer, "param_names"):
                        self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                probs = getattr(self.loss, "y_pred", y_pred)
                pred_labels, true_labels = self._predicted_and_true_labels(
                    probs, y_batch
                )
                epoch_acc += np.mean(pred_labels == true_labels)

            train_loss = epoch_loss / batches
            train_acc = epoch_acc / batches
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            if X_val is not None:
                y_pred_val = self.forward(X_val)
                val_loss = self.loss.forward(y_pred_val, y_val)

                probs_val = getattr(self.loss, "y_pred", y_pred_val)
                pred_labels_val, true_labels_val = self._predicted_and_true_labels(
                    probs_val, y_val
                )
                val_acc = np.mean(pred_labels_val == true_labels_val)

                val_losses.append(val_loss)
                val_accs.append(val_acc)

            if verbose and epoch % 1 == 0:
                print(
                    f"epoch {epoch+1}: training: loss={train_loss:.6f}, "
                    f"acc={train_acc:.6f}, lr={self.optimizer.current_learning_rate}"
                )
                if X_val is not None:
                    print(
                        f"epoch {epoch+1}: validation: loss={val_loss:.6f}, acc={val_acc:.6f}"
                    )
                print()

        # Plot training and validation curves
        epochs_range = range(1, epochs + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, train_losses, label="train loss")
        if X_val is not None:
            plt.plot(epochs_range, val_losses, label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs Epoch")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, train_accs, label="train acc")
        if X_val is not None:
            plt.plot(epochs_range, val_accs, label="val acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Epoch")
        plt.legend()
        plt.grid(True)
        plt.show()

        return {
            "train_loss": train_losses,
            "train_acc": train_accs,
            "val_loss": val_losses,
            "val_acc": val_accs,
        }
