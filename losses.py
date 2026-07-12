"""
Optimizers.

All optimizers share the same 3-method contract that Model.fit() calls
once per batch:
    pre_update_params()          -> recompute current_learning_rate (decay)
    update_params(layer)         -> mutate every one of the layer's
                                     parameters in place, using the
                                     gradients the layer's own backward()
                                     computed
    post_update_params()         -> bookkeeping (increment iteration count)

Generalized vs. ann_framework/cnn_framework: those frameworks only ever
had one weight matrix + one bias per trainable layer, so update_params()
could just hardcode `layer.w -= lr * layer.dw` (and the same for `.b`).
Layer_LSTM alone has 4 gates, each with its own input weights, recurrent
weights, and bias - 12 separate arrays. Rather than special-casing every
layer type inside every optimizer, every trainable layer here declares
its own `layer.param_names` list, e.g.:

    Layer_Dense.param_names     = ["w", "b"]
    Layer_Embedding.param_names = ["w"]
    Layer_SimpleRNN.param_names = ["Wx", "Wh", "b"]
    Layer_LSTM.param_names      = ["Wxf", "Whf", "bf", "Wxi", "Whi", "bi",
                                    "Wxo", "Who", "bo", "Wxg", "Whg", "bg"]
    Layer_GRU.param_names       = ["Wxz", "Whz", "bz", "Wxr", "Whr", "br",
                                    "Wxn", "Whn", "bn"]

and every optimizer below just loops over `layer.param_names`, reading
`getattr(layer, name)` / `getattr(layer, "d" + name)` instead of
hardcoding attribute names - the exact same momentum / Adagrad / RMSprop
/ Adam math applies unchanged, just per named parameter instead of
hardcoded per w/b. Per-parameter optimizer state (momentum velocity,
squared-gradient cache) is stored back on the layer as
`layer.<name>_velocity` / `layer.<name>_cache`, lazily created on first
use - same idea as before, just named generically.
"""

import numpy as np


class Optimizer_normalGD:
    """Plain (vanilla) gradient descent with a fixed learning rate."""

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        # kept for interface consistency with the other optimizers (e.g.
        # Model.fit()'s progress logging reads current_learning_rate)
        self.current_learning_rate = learning_rate

    def pre_update_params(self):
        return

    def update_params(self, layer):
        for name in layer.param_names:
            param = getattr(layer, name)
            grad = getattr(layer, "d" + name)
            param -= self.learning_rate * grad

    def post_update_params(self):
        return


class Optimizer_GD_decay:
    """Gradient descent with 1/(1 + decay*t) learning-rate decay."""

    def __init__(self, learning_rate=1, decay=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (
                1 + self.decay * self.iterations
            )

    def update_params(self, layer):
        for name in layer.param_names:
            param = getattr(layer, name)
            grad = getattr(layer, "d" + name)
            param -= self.current_learning_rate * grad

    def post_update_params(self):
        self.iterations += 1


class Optimizer_momentum:
    """Gradient descent with (optional) momentum and learning-rate decay."""

    def __init__(self, learning_rate=1, decay=0, momentum=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (
                1 + self.decay * self.iterations
            )

    def update_params(self, layer):
        for name in layer.param_names:
            param = getattr(layer, name)
            grad = getattr(layer, "d" + name)

            if self.momentum:
                # hasattr() checks whether an object has a particular
                # attribute/method - used here for lazy-initializing the
                # momentum buffer for this parameter on first use.
                velocity_name = name + "_velocity"
                if not hasattr(layer, velocity_name):
                    setattr(layer, velocity_name, np.zeros_like(param))

                velocity = self.momentum * getattr(layer, velocity_name) - (
                    self.current_learning_rate * grad
                )
                setattr(layer, velocity_name, velocity)
                update = velocity
            else:
                update = -self.current_learning_rate * grad

            param += update

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adagrad:
    """Adagrad: per-parameter learning rate scaled by the running sum of squared gradients."""

    def __init__(self, learning_rate=1, decay=0, epsilion=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilion = epsilion

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (
                1 + self.decay * self.iterations
            )

    def update_params(self, layer):
        for name in layer.param_names:
            param = getattr(layer, name)
            grad = getattr(layer, "d" + name)

            cache_name = name + "_cache"
            if not hasattr(layer, cache_name):
                setattr(layer, cache_name, np.zeros_like(param))

            cache = getattr(layer, cache_name) + grad ** 2
            setattr(layer, cache_name, cache)

            param -= (
                self.current_learning_rate * grad / (np.sqrt(cache) + self.epsilion)
            )

    def post_update_params(self):
        self.iterations += 1


class Optimizer_RMSprop:
    """
    RMSprop: like Adagrad, but the cache is an exponentially decaying
    moving average of squared gradients (controlled by `rho`) instead of
    an ever-growing sum, so the effective learning rate doesn't vanish.

    Hyperparameter notes:
      - learning_rate: usually much smaller than for plain SGD; 0.001-0.01
        is a common starting point (values like 1 tend to diverge).
      - rho: how much past gradient history is kept. Close to 1 (0.9-0.99)
        gives smooth, stable updates; rho=0 reduces RMSprop to reacting
        only to the current gradient; rho=1 freezes the cache.
      - epsilion: tiny constant (1e-7 to 1e-8) to avoid division by zero,
        rarely tuned.
      - decay: optional learning-rate decay over iterations, often 0
        initially.
      A reasonable default is learning_rate=0.001, rho=0.9, epsilion=1e-7,
      decay=0.
    """

    def __init__(self, learning_rate=1, decay=0, epsilion=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilion = epsilion
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (
                1 + self.decay * self.iterations
            )

    def update_params(self, layer):
        for name in layer.param_names:
            param = getattr(layer, name)
            grad = getattr(layer, "d" + name)

            cache_name = name + "_cache"
            if not hasattr(layer, cache_name):
                setattr(layer, cache_name, np.zeros_like(param))

            cache = (
                self.rho * getattr(layer, cache_name) + (1 - self.rho) * grad ** 2
            )
            setattr(layer, cache_name, cache)

            param -= (
                self.current_learning_rate * grad / (np.sqrt(cache) + self.epsilion)
            )

    def post_update_params(self):
        self.iterations += 1


class Optimizer_ADAM:
    """Adam: momentum (first moment) + RMSprop-style cache (second moment), both bias-corrected."""

    def __init__(
        self,
        learning_rate=0.001,
        decay=0,
        epsilion=1e-7,
        beta_1=0.9,
        beta_2=0.999,
    ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilion = epsilion
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (
                1 + self.decay * self.iterations
            )

    def update_params(self, layer):
        for name in layer.param_names:
            param = getattr(layer, name)
            grad = getattr(layer, "d" + name)

            momentum_name = name + "_momentum"
            cache_name = name + "_cache"
            if not hasattr(layer, cache_name):
                setattr(layer, momentum_name, np.zeros_like(param))
                setattr(layer, cache_name, np.zeros_like(param))

            # first moment (momentum)
            momentum = (
                self.beta_1 * getattr(layer, momentum_name)
                + (1 - self.beta_1) * grad
            )
            setattr(layer, momentum_name, momentum)
            momentum_corrected = momentum / (
                1 - self.beta_1 ** (self.iterations + 1)
            )

            # second moment (cache)
            cache = (
                self.beta_2 * getattr(layer, cache_name)
                + (1 - self.beta_2) * grad ** 2
            )
            setattr(layer, cache_name, cache)
            cache_corrected = cache / (1 - self.beta_2 ** (self.iterations + 1))

            param -= (
                self.current_learning_rate
                * momentum_corrected
                / (np.sqrt(cache_corrected) + self.epsilion)
            )

    def post_update_params(self):
        self.iterations += 1
