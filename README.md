# rnn_framework_from_scratch

A recurrent neural network framework built **from scratch using only NumPy** —
no PyTorch, no TensorFlow, no autograd. Every forward pass, every backward
pass through time (BPTT), and every parameter update is derived and
implemented by hand.

This project exists to demonstrate a real, working understanding of how
recurrent networks actually compute — not just how to call `.fit()` on a
library. It's the sequence-model companion to two earlier from-scratch
projects:

> **[ann_framework_from_scratch](https://github.com/<your-username>/ann_framework_from_scratch)** —
> feed-forward networks (Dense layers, activations, losses, optimizers).
>
> **[cnn_framework_from_scratch](https://github.com/<your-username>/cnn_framework_from_scratch)** —
> convolutional networks (Conv2D/Conv3D, MaxPool2D, Flatten).

This repo reuses the exact same activations/losses/optimizers/`Model`
machinery and coding style as those two, and adds what sequence models
need on top: an `Embedding` layer and three recurrent cells (`SimpleRNN`,
`LSTM`, `GRU`), built around a concrete target task — **sentiment
analysis**, i.e. "read a whole sequence of inputs, output one single
prediction" (many-to-one).

---

## Table of contents

- [What's implemented](#whats-implemented)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Why many-to-one](#why-many-to-one)
- [API overview](#api-overview)
  - [Activations](#activations)
  - [Losses](#losses)
  - [Layers](#layers)
  - [Optimizers (generalized)](#optimizers-generalized)
  - [Model](#model)
- [How this differs from ann_framework / cnn_framework](#how-this-differs-from-ann_framework--cnn_framework)
- [Design notes: the math behind each cell](#design-notes-the-math-behind-each-cell)
- [Example: sentiment classification](#example-sentiment-classification)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [License](#license)

---

## What's implemented

| Category | Components |
|---|---|
| **Activations** | ReLU, Leaky ReLU, Sigmoid, **Tanh (new)**, Softmax |
| **Losses** | Categorical Cross-Entropy, fused Softmax+CCE, Mean Squared Error, **Binary Cross-Entropy (new)**, **fused Sigmoid+BCE (new)** |
| **Optimizers** | SGD, SGD with learning-rate decay, SGD with Momentum, Adagrad, RMSprop, Adam — **generalized to work with any number of weight matrices per layer** |
| **Layers** | Dense (fully-connected), **Embedding (new)**, **SimpleRNN (new)**, **LSTM (new)**, **GRU (new)** — the three recurrent cells are all many-to-one |
| **Training** | The same Keras-like `Model` class with `compile()` / `fit()`, mini-batch training, automatic train/validation split, loss & accuracy curves |

Every layer/activation/loss was implemented with an explicit `forward()`
and `backward()` method, derived by hand from the underlying calculus —
this README explains the math behind each recurrent cell below.

---

## Project structure

```
rnn_framework_from_scratch/
├── rnn_framework/
│   ├── __init__.py           # public API exports
│   ├── activations.py        # ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
│   ├── losses.py              # CCE, fused Softmax+CCE, Squared Loss, BCE, fused Sigmoid+BCE
│   ├── optimizers.py          # SGD, Momentum, Adagrad, RMSprop, Adam (generalized)
│   ├── layers_dense.py        # Layer_Dense
│   ├── layers_embedding.py    # Layer_Embedding
│   ├── layers_recurrent.py    # Layer_SimpleRNN, Layer_LSTM, Layer_GRU
│   └── model.py               # Model (compile / forward / backward / fit)
├── examples/
│   └── sentiment_classification.py
├── tests/
│   └── test_rnn_framework.py   # unit tests incl. numerical gradient checks
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/rnn_framework_from_scratch.git
cd rnn_framework_from_scratch
pip install -r requirements.txt

# optional: install as an editable package so `import rnn_framework`
# works from anywhere
pip install -e .
```

Requires Python 3.8+ and NumPy. Matplotlib is used for the built-in
loss/accuracy plots in `Model.fit()`. Nothing else — the sentiment
example builds its own tiny synthetic dataset in-file, so there's no
dataset download involved.

---

## Quick start

```python
import numpy as np
from rnn_framework import (
    Layer_Embedding, Layer_LSTM, Layer_Dense, Activation_Sigmoid,
    Loss_BinaryCrossentropy, Optimizer_ADAM, Model,
)

vocab_size = 500
seq_len = 10

X = np.random.randint(0, vocab_size, size=(300, seq_len))  # token ids
y = np.random.randint(0, 2, size=300)                       # 0/1 sentiment

model = Model([
    Layer_Embedding(vocab_size, embedding_dim=16),
    Layer_LSTM(32),
    Layer_Dense(1),
    Activation_Sigmoid(),
])

model.compile(
    loss=Loss_BinaryCrossentropy(),
    optimizer=Optimizer_ADAM(learning_rate=0.01),
)

model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)
```

Swap `Layer_LSTM(32)` for `Layer_SimpleRNN(32)` or `Layer_GRU(32)` and
nothing else changes — all three recurrent cells share the same
`(N, T, input_dim) -> (N, n_units)` interface.

---

## Why many-to-one

A sequence model can be many things: many-to-many (translation), one-to-many
(image captioning), many-to-one (classification of a whole sequence). This
repo picks **many-to-one on purpose**, because that's what the target task —
sentiment analysis — actually is: **multiple inputs** (every word/token in
the review) collapse down to **a single output** (positive or negative).
None of the three recurrent layers here have a `return_sequences` option to
expose the hidden state at every timestep; they only ever return the final
hidden state, `h_{T-1}`. That keeps the implementation focused on exactly
the case this repo is for, instead of a more general (and more
complicated) layer nobody asked for.

---

## API overview

### Activations

All activations implement the same 2-method contract:

```python
output   = activation.forward(inputs)
d_inputs = activation.backward(d_output)   # gradient w.r.t. inputs
```

| Class | Forward | Backward |
|---|---|---|
| `Activation_ReLU` | `max(0, x)` | passes gradient through where `x > 0`, zero elsewhere |
| `Activation_LeakyReLU(alpha=0.01)` | `max(alpha*x, x)` | scales gradient by `alpha` where `x <= 0` |
| `Activation_Sigmoid` | `1 / (1 + e^-x)` | `d_output * sigmoid(x) * (1 - sigmoid(x))` |
| `Activation_Tanh` **(new)** | `tanh(x)` | `d_output * (1 - tanh(x)^2)` |
| `Activation_Softmax` | numerically-stable row-wise softmax | full Jacobian-vector product |

`Activation_Tanh` is new in this repo — it's the squashing function every
recurrent cell below uses *internally*. Note that `Layer_SimpleRNN` /
`Layer_LSTM` / `Layer_GRU` don't actually call this class; the tanh math is
inlined directly into their own `forward()`/`backward()`, because BPTT
needs the cached hidden/cell states at *every* timestep, not just one
layer's input/output. `Activation_Tanh` is provided as a standalone building
block for stacking in a plain feed-forward `Model` if you ever want one.

### Losses

| Class | Purpose |
|---|---|
| `Loss_CategoricalCrossentropy` | multi-class classification; sparse `(N,)` or one-hot `(N, C)` labels |
| `Activation_Softmax_Loss_CategoricalCrossentropy` | softmax + CCE, fused |
| `Squared_Loss` | mean squared error, for regression-style targets |
| `Loss_BinaryCrossentropy` **(new)** | single-output (two-class) classification; `y_pred` shape `(N, 1)` |
| `Activation_Sigmoid_Loss_BinaryCrossentropy` **(new)** | sigmoid + BCE, fused |

**Why binary cross-entropy at all, when categorical cross-entropy already
exists?** Sentiment analysis (this repo's target task) is a two-class
problem, and modelling two classes with one sigmoid neuron (`P(positive)`)
is simpler and cheaper than modelling them with a two-neuron softmax. BCE
is the matching loss for that one-neuron-probability setup:

```
L = -mean( y*log(p) + (1-y)*log(1-p) )
```

**Why fuse sigmoid + BCE?** Exactly the same reasoning as the existing
softmax+CCE fusion: differentiating sigmoid and then BCE separately
involves dividing by `p*(1-p)`, which blows up as `p` approaches 0 or 1.
Fusing collapses the combined gradient to:

```
dL/dz = (y_pred - y_true) / batch_size
```

where `z` is the raw logit fed into the sigmoid — cheap and numerically
stable, mirroring the softmax+CCE fusion exactly. `Model.compile()`
auto-detects this pattern the same way it already detects softmax+CCE: if
the last layer is `Activation_Sigmoid` and the loss is
`Loss_BinaryCrossentropy`, it swaps in the fused class and pops the
standalone sigmoid off the layer list.

> **Watch out at inference time:** because `compile()` pops the standalone
> `Activation_Sigmoid`/`Activation_Softmax` layer, `model.forward(X)` after
> compiling returns **raw logits**, not probabilities — the activation only
> gets applied inside the fused loss during training. To get an actual
> probability back out after training, run the logits through a fresh
> `Activation_Sigmoid()` (or `Activation_Softmax()`) yourself. See
> `examples/sentiment_classification.py`'s `predict_sentiment()` for exactly
> this.

### Layers

**`Layer_Dense(n_neurons)`** — unchanged from ann_framework/cnn_framework:
lazy weight initialization, `output = inputs @ W + b`.

**`Layer_Embedding(vocab_size, embedding_dim)`** **(new)**

A trainable lookup table, shape `(vocab_size, embedding_dim)`. This is
normally the first layer in a text model — raw text gets turned into
integer token ids *outside* this framework (e.g. a plain `word -> index`
dictionary), and `Layer_Embedding` turns those ids into vectors a
recurrent layer can do math on.

```
forward:   inputs (N, T) int ids  ->  output (N, T, embedding_dim)
                                       (self.w[inputs], fancy indexing)
backward:  d_output (N, T, embedding_dim)  ->  dw accumulated via
                                                np.add.at(dw, inputs, d_output)
```

`np.add.at` matters here: a token appearing more than once in a batch (very
common - "the", "was", ...) needs its gradient **accumulated** from every
occurrence, not overwritten. Plain `dw[inputs] += d_output` silently drops
all but one occurrence for repeated indices; `np.add.at` doesn't. Unlike
`Layer_Dense`, `vocab_size` is known upfront, so the embedding table is
initialized eagerly in `__init__` rather than lazily on first `forward()`.
There's no `d_inputs` to propagate further back — token ids aren't
differentiable, so this layer is only ever meant to go first.

**`Layer_SimpleRNN(n_units)` / `Layer_LSTM(n_units)` / `Layer_GRU(n_units)`** **(new)**

All three share one interface:

```
forward:   inputs (N, T, input_dim)  ->  output (N, n_units)
                                          (hidden state after the last timestep)
backward:  d_output (N, n_units)  ->  d_inputs (N, T, input_dim)
```

Lazy weight initialization, same pattern as `Layer_Dense`: you only pass
`n_units`; `input_dim` is inferred from the first batch's shape. See
[Design notes](#design-notes-the-math-behind-each-cell) below for the full
forward/backward derivation of each cell.

### Optimizers (generalized)

Same 6 optimizers as before (plain GD, GD+decay, momentum, Adagrad,
RMSprop, Adam), sharing the same 3-method contract:

```python
optimizer.pre_update_params()     # recompute the decayed learning rate
optimizer.update_params(layer)    # mutate every one of the layer's parameters in place
optimizer.post_update_params()    # increment the internal iteration count
```

The one real change in this repo is *inside* `update_params()` — see
[How this differs](#how-this-differs-from-ann_framework--cnn_framework)
below for why and how.

### Model

Same `Model(layers)` / `compile()` / `forward()` / `backward()` / `fit()`
as before — mini-batch training with automatic train/validation split,
per-epoch loss/accuracy logging, and Matplotlib loss/accuracy curves at
the end. Two small additions:

- `compile()` also fuses `Activation_Sigmoid` + `Loss_BinaryCrossentropy`
  (see [Losses](#losses) above), in addition to the existing
  Softmax+CCE fusion.
- The accuracy calculation in `fit()` now branches on the output shape:
  `argmax` for a softmax `(N, classes)` output, threshold-at-0.5 for a
  sigmoid `(N, 1)` output.

---

## How this differs from ann_framework / cnn_framework

Everything not mentioned above (`Layer_Dense`, `Activation_ReLU`/`Sigmoid`/
`Softmax`, `Loss_CategoricalCrossentropy`, the fused Softmax+CCE loss,
`Squared_Loss`, the overall shape of `Model`) is copied over unchanged. Two
things had to change to fit recurrent cells in, though:

**1. The optimizer contract used to hardcode `.w` / `.b`.**
In ann_framework/cnn_framework, every trainable layer had exactly one
weight matrix and one bias vector, so every optimizer's `update_params()`
could just hardcode `layer.w -= lr * layer.dw` (and the same for `.b`).
`Layer_LSTM` alone has **4 gates**, each with its own input weights,
recurrent weights, and bias (`Wxf`/`Whf`/`bf`, `Wxi`/`Whi`/`bi`, ...) - 12
separate arrays. Rather than special-casing every layer type inside every
optimizer, every layer here declares its own `layer.param_names` list:

```python
Layer_Dense.param_names     = ["w", "b"]
Layer_Embedding.param_names = ["w"]
Layer_SimpleRNN.param_names = ["Wx", "Wh", "b"]
Layer_LSTM.param_names      = ["Wxf", "Whf", "bf", "Wxi", "Whi", "bi",
                                "Wxo", "Who", "bo", "Wxg", "Whg", "bg"]
Layer_GRU.param_names       = ["Wxz", "Whz", "bz", "Wxr", "Whr", "br",
                                "Wxn", "Whn", "bn"]
```

Every optimizer's `update_params(layer)` just loops over
`layer.param_names`, reading `getattr(layer, name)` / `getattr(layer, "d"
+ name)` — the exact same momentum / Adagrad / RMSprop / Adam math applies
unchanged, just per named parameter instead of hardcoded per w/b.
Per-parameter optimizer state (momentum velocity, squared-gradient cache)
is stored back on the layer as `layer.<name>_velocity` /
`layer.<name>_cache`, lazily created on first use — same idea as
`previous_weight_updates` / `weight_cache` before, just named generically.

**2. `Model.fit()` used to find trainable layers by `isinstance()`.**
`isinstance(layer, (Layer_Dense, Layer_Conv2D, Layer_Conv3D))` would need
a new entry every time a layer type is added — and this repo adds four
(`Layer_Embedding`, `Layer_SimpleRNN`, `Layer_LSTM`, `Layer_GRU`). Instead,
`fit()` now checks `hasattr(layer, "param_names")` — any layer that
declares that attribute is automatically picked up as trainable, no edits
to `Model` needed for future layer types either.

---

## Design notes: the math behind each cell

### SimpleRNN

```
h_t = tanh(x_t @ Wx + h_{t-1} @ Wh + b)          for t = 0 .. T-1
h_{-1} = 0
output = h_{T-1}
```

The simplest possible recurrent cell: one input weight matrix, one
recurrent weight matrix, one bias, one tanh. It's also the most prone to
vanishing/exploding gradients over long sequences — repeatedly
multiplying by `Wh` and passing through `tanh'` (which is at most 1, and
usually much less) shrinks gradients geometrically the further back in
time they travel. That's exactly the problem LSTM and GRU are designed to
fix.

Backward pass (BPTT): `dh_next` carries `dL/dh_t` backwards across
timesteps, starting as the incoming `d_output` (gradient at the last
timestep) and, at every earlier timestep, becoming "gradient contributed
by being used as `h_{t-1}` in the next step's recurrence" — computed as
`dz @ Wh.T` where `dz` is the gradient through that timestep's tanh.

### LSTM

```
f_t = sigmoid(x_t @ Wxf + h_{t-1} @ Whf + bf)   # forget gate
i_t = sigmoid(x_t @ Wxi + h_{t-1} @ Whi + bi)   # input gate
o_t = sigmoid(x_t @ Wxo + h_{t-1} @ Who + bo)   # output gate
g_t = tanh(x_t @ Wxg + h_{t-1} @ Whg + bg)      # candidate cell state
c_t = f_t * c_{t-1} + i_t * g_t                  # new cell state
h_t = o_t * tanh(c_t)                            # new hidden state
h_{-1} = c_{-1} = 0
output = h_{T-1}
```

Each gate keeps its own separate weight matrices rather than one big
concatenated matrix — more names to track, but every matrix's shape and
role maps directly onto the equations above, matching the explicit style
used everywhere else in this project (e.g. `Layer_Conv2D` keeps its
padding/stride math just as spelled-out).

The forget/input/output gates decide, at every timestep, how much of the
old cell state to keep, how much new candidate information to write in,
and how much of the cell state to expose as the hidden state. Critically,
`c_t = f_t * c_{t-1} + i_t * g_t` lets gradient flow from `c_t` back to
`c_{t-1}` through simple multiplication by `f_t` — no repeated
matrix-multiply-then-squash like SimpleRNN's `h_{t-1} @ Wh` — which is
the "gradient highway" that lets LSTM learn dependencies across far more
timesteps before vanishing.

Backward pass carries **two** running gradients backwards, `dh_next` and
`dc_next` (`dc_next` starts at zero, since nothing after the last timestep
uses `c_{T-1}` directly - only through `h_{T-1}`). At every timestep:

```
do_t = dh * tanh(c_t)                          # h_t = o_t * tanh(c_t)
dc_t = dh * o_t * (1 - tanh(c_t)^2) + dc_next
df_t = dc_t * c_{t-1}                          # c_t = f_t*c_{t-1} + i_t*g_t
di_t = dc_t * g_t
dg_t = dc_t * i_t
dc_prev = dc_t * f_t                            # gradient straight into c_{t-1}
```

then each gate's raw pre-activation gradient is obtained by multiplying
through its own nonlinearity's derivative (`sigmoid'` for f/i/o,
`tanh'` for g), and those raw gradients are what actually get matmul'd
against `Wx*`/`Wh*` to build the weight gradients and `dh_prev`/`dx_t`.

### GRU

```
z_t = sigmoid(x_t @ Wxz + h_{t-1} @ Whz + bz)                # update gate
r_t = sigmoid(x_t @ Wxr + h_{t-1} @ Whr + br)                # reset gate
n_t = tanh(x_t @ Wxn + (r_t * h_{t-1}) @ Whn + bn)           # candidate hidden state
h_t = (1 - z_t) * h_{t-1} + z_t * n_t
h_{-1} = 0
output = h_{T-1}
```

GRU has no separate cell state — it merges LSTM's forget/input gates into
a single update gate `z_t` (`z_t` near 1 means "mostly use the new
candidate", near 0 means "mostly keep the old hidden state"), and the
reset gate `r_t` controls how much of the previous hidden state is
allowed to influence the new candidate `n_t`. Fewer gates than LSTM (2
instead of 3, plus the candidate), so fewer parameters, and often trains
just as well in practice.

Backward pass: `h_t = (1-z_t)*h_{t-1} + z_t*n_t` gives two direct paths
back to `h_{t-1}` - one through the `(1-z_t)` term
(`dh_prev_direct = dh * (1 - z_t)`), one indirectly through `n_t`'s
dependence on `r_t * h_{t-1}` via `Whn` (`dh_prev_via_r`). Both get summed
together (along with the `z_t`/`r_t` gate contributions through `Whz`/
`Whr`) into the single `dh_prev` carried to the previous timestep -
mirroring how LSTM sums multiple gate contributions into its `dh_prev`.

**All three cells' backward passes were verified against central-difference
numerical gradients for every single weight matrix and bias (not just
`d_inputs`)** — see `tests/test_rnn_framework.py`.

---

## Example: sentiment classification

`examples/sentiment_classification.py` builds a tiny synthetic dataset
in-file (short "reviews" made of neutral filler words plus one planted
positive or negative sentiment word at a random position), trains
`Embedding -> LSTM -> Dense -> Sigmoid` on it, and evaluates it on a few
brand-new sentences.

```bash
pip install -r requirements.txt
python examples/sentiment_classification.py
```

Swap `Layer_LSTM(16)` for `Layer_SimpleRNN(16)` or `Layer_GRU(16)` to
compare cells on the exact same task — nothing else in the script needs
to change.

---

## Testing

The test suite (`tests/test_rnn_framework.py`) covers:

- Shape and value correctness for `Activation_Tanh`, `Loss_BinaryCrossentropy`,
  and `Layer_Embedding` (including gradient accumulation for repeated
  tokens).
- **Numerical gradient checks** (finite-difference vs. analytic) for
  `Activation_Tanh`, the fused Sigmoid+BCE loss, `Layer_Embedding`, and —
  most importantly — **every single weight matrix and bias of
  `Layer_SimpleRNN`, `Layer_LSTM`, and `Layer_GRU`**, plus their input
  gradients. This is the standard way to verify a hand-derived BPTT
  backward pass is actually correct, not just shape-compatible.
- The generalized optimizer contract (`param_names` on `Layer_Dense` vs.
  `Layer_LSTM`, an Adam update actually reducing a toy loss).
- `Model.compile()`'s automatic Sigmoid+BCE fusion (and that Softmax+CCE
  fusion still works).
- An end-to-end sentiment training run, parametrized across all three
  recurrent cells, asserting the loss decreases and validation accuracy
  ends up reasonably high.

```bash
pip install pytest
pytest tests/
```

---

## Known limitations

This is an educational/from-scratch implementation, not a
production-grade library:

- No GPU support — everything runs on NumPy (CPU), and BPTT loops over
  timesteps in plain Python, so long sequences or large hidden sizes will
  be noticeably slower than a vectorized/GPU framework.
- No autograd — every `backward()` is a manually-derived, layer-specific
  gradient, so adding a new layer means deriving its BPTT gradient by
  hand.
- Many-to-one only — there's no `return_sequences` option to get the
  hidden state at every timestep out of `Layer_SimpleRNN`/`Layer_LSTM`/
  `Layer_GRU`, and no stacking/bidirectional variants. That's a deliberate
  scope choice (see [Why many-to-one](#why-many-to-one)), not an
  oversight — the goal was to do the sentiment-analysis case properly,
  not to rebuild every RNN variant.
- No gradient clipping — long sequences with a poorly-scaled learning
  rate can still blow up, especially with `Layer_SimpleRNN`. Keep an eye
  on the loss curve and lower the learning rate if it diverges.
- `Model.fit()`'s progress plots always call `matplotlib.pyplot.show()`,
  which blocks in non-interactive environments unless you set a
  non-interactive backend (e.g. `matplotlib.use("Agg")`) first.

---

## License

MIT — see [LICENSE](LICENSE).
