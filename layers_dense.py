"""
Recurrent layers (SimpleRNN, LSTM, GRU), implemented from scratch with
NumPy. All three are many-to-one on purpose: they read a whole sequence
of inputs and only ever return the final hidden state, which is exactly
what the target task (sentiment analysis) needs - see the README's "Why
many-to-one" section. Every forward pass loops over timesteps and caches
everything backward() needs; every backward pass is a hand-derived BPTT
(backprop through time), not autograd.

All three share one interface:
    forward:   inputs (N, T, input_dim)  ->  output (N, n_units)
    backward:  d_output (N, n_units)     ->  d_inputs (N, T, input_dim)

Lazy weight initialization, same pattern as Layer_Dense: you only pass
n_units, input_dim is inferred from the first forward() call.
"""

import numpy as np


def sigmoid(x):
    """
    Plain sigmoid helper, used internally by Layer_LSTM/Layer_GRU's gates.

    This is intentionally a bare function and not Activation_Sigmoid -
    the gate math below is inlined directly into forward()/backward()
    because BPTT needs the cached gate values at *every* timestep, not
    just one layer's input/output like Activation_Sigmoid caches.
    """
    return 1 / (1 + np.exp(-x))


class Layer_SimpleRNN:
    """
    Vanilla recurrent cell:

        h_t = tanh(x_t @ Wx + h_{t-1} @ Wh + b)      for t = 0 .. T-1
        h_{-1} = 0
        output = h_{T-1}

    The simplest possible recurrent cell: one input weight matrix, one
    recurrent weight matrix, one bias, one tanh. It's also the most prone
    to vanishing/exploding gradients over long sequences - repeatedly
    multiplying by Wh and passing through tanh' (which is at most 1, and
    usually much less) shrinks gradients geometrically the further back
    in time they travel. That's exactly the problem LSTM and GRU below
    are designed to fix.
    """

    param_names = ["Wx", "Wh", "b"]

    def __init__(self, n_units):
        self.n_units = n_units
        self.initialized = False

    def forward(self, inputs):
        if not self.initialized:
            # -----
            input_dim = inputs.shape[2]
            self.Wx = 0.01 * np.random.randn(input_dim, self.n_units)
            self.Wh = 0.01 * np.random.randn(self.n_units, self.n_units)
            self.b = np.zeros((1, self.n_units))
            self.initialized = True
            # -----
            # lazy initialization, same idea as Layer_Dense: input_dim is
            # only known once we see the first batch

        self.inputs = inputs
        N, T, input_dim = inputs.shape

        # cache the hidden state at every timestep - BPTT needs all of
        # them, not just the final one. h[0] is h_{-1} = 0 (already
        # zeros from np.zeros); h[t + 1] is h_t.
        self.h = np.zeros((T + 1, N, self.n_units))

        for t in range(T):
            x_t = inputs[:, t, :]
            z_t = np.dot(x_t, self.Wx) + np.dot(self.h[t], self.Wh) + self.b
            self.h[t + 1] = np.tanh(z_t)

        self.output = self.h[T]
        return self.output

    def backward(self, d_output):
        N, T, input_dim = self.inputs.shape

        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.db = np.zeros_like(self.b)
        self.d_inputs = np.zeros_like(self.inputs)

        # dh_next carries dL/dh_t backwards across timesteps - starts as
        # the incoming gradient at the last timestep (h_{T-1} = output),
        # and at every earlier timestep becomes "gradient contributed by
        # being used as h_{t-1} in the next step's recurrence".
        dh_next = d_output

        for t in reversed(range(T)):
            x_t = self.inputs[:, t, :]
            h_prev = self.h[t]
            h_t = self.h[t + 1]

            # dz_t = dh_t * tanh'(z_t) = dh_t * (1 - h_t^2), since
            # h_t = tanh(z_t)
            dz_t = dh_next * (1 - h_t ** 2)

            self.dWx += np.dot(x_t.T, dz_t)
            self.dWh += np.dot(h_prev.T, dz_t)
            self.db += np.sum(dz_t, axis=0, keepdims=True)

            self.d_inputs[:, t, :] = np.dot(dz_t, self.Wx.T)

            # gradient contributed by h_prev being used as h_{t-1} in
            # this step's recurrence, carried back to the previous
            # timestep
            dh_next = np.dot(dz_t, self.Wh.T)

        return self.d_inputs


class Layer_LSTM:
    """
    LSTM cell:

        f_t = sigmoid(x_t @ Wxf + h_{t-1} @ Whf + bf)   # forget gate
        i_t = sigmoid(x_t @ Wxi + h_{t-1} @ Whi + bi)   # input gate
        o_t = sigmoid(x_t @ Wxo + h_{t-1} @ Who + bo)   # output gate
        g_t = tanh(x_t @ Wxg + h_{t-1} @ Whg + bg)      # candidate cell state
        c_t = f_t * c_{t-1} + i_t * g_t                  # new cell state
        h_t = o_t * tanh(c_t)                            # new hidden state
        h_{-1} = c_{-1} = 0
        output = h_{T-1}

    Each gate keeps its own separate weight matrices rather than one big
    concatenated matrix - more names to track, but every matrix's shape
    and role maps directly onto the equations above, matching the
    explicit style used everywhere else in this project.

    The forget/input/output gates decide, at every timestep, how much of
    the old cell state to keep, how much new candidate information to
    write in, and how much of the cell state to expose as the hidden
    state. Critically, c_t = f_t * c_{t-1} + i_t * g_t lets gradient flow
    from c_t back to c_{t-1} through simple multiplication by f_t - no
    repeated matrix-multiply-then-squash like SimpleRNN's h_{t-1} @ Wh -
    which is the "gradient highway" that lets LSTM learn dependencies
    across far more timesteps before vanishing.
    """

    param_names = [
        "Wxf", "Whf", "bf",
        "Wxi", "Whi", "bi",
        "Wxo", "Who", "bo",
        "Wxg", "Whg", "bg",
    ]

    def __init__(self, n_units):
        self.n_units = n_units
        self.initialized = False

    def forward(self, inputs):
        if not self.initialized:
            input_dim = inputs.shape[2]
            n = self.n_units

            self.Wxf = 0.01 * np.random.randn(input_dim, n)
            self.Whf = 0.01 * np.random.randn(n, n)
            self.bf = np.zeros((1, n))

            self.Wxi = 0.01 * np.random.randn(input_dim, n)
            self.Whi = 0.01 * np.random.randn(n, n)
            self.bi = np.zeros((1, n))

            self.Wxo = 0.01 * np.random.randn(input_dim, n)
            self.Who = 0.01 * np.random.randn(n, n)
            self.bo = np.zeros((1, n))

            self.Wxg = 0.01 * np.random.randn(input_dim, n)
            self.Whg = 0.01 * np.random.randn(n, n)
            self.bg = np.zeros((1, n))

            self.initialized = True

        self.inputs = inputs
        N, T, input_dim = inputs.shape
        n = self.n_units

        # cache every timestep's gates/states - BPTT needs all of them.
        # h[0] = h_{-1} = 0, c[0] = c_{-1} = 0 (already zeros)
        self.h = np.zeros((T + 1, N, n))
        self.c = np.zeros((T + 1, N, n))
        self.f = np.zeros((T, N, n))
        self.i = np.zeros((T, N, n))
        self.o = np.zeros((T, N, n))
        self.g = np.zeros((T, N, n))

        for t in range(T):
            x_t = inputs[:, t, :]
            h_prev = self.h[t]
            c_prev = self.c[t]

            f_t = sigmoid(np.dot(x_t, self.Wxf) + np.dot(h_prev, self.Whf) + self.bf)
            i_t = sigmoid(np.dot(x_t, self.Wxi) + np.dot(h_prev, self.Whi) + self.bi)
            o_t = sigmoid(np.dot(x_t, self.Wxo) + np.dot(h_prev, self.Who) + self.bo)
            g_t = np.tanh(np.dot(x_t, self.Wxg) + np.dot(h_prev, self.Whg) + self.bg)

            c_t = f_t * c_prev + i_t * g_t
            h_t = o_t * np.tanh(c_t)

            self.f[t] = f_t
            self.i[t] = i_t
            self.o[t] = o_t
            self.g[t] = g_t
            self.c[t + 1] = c_t
            self.h[t + 1] = h_t

        self.output = self.h[T]
        return self.output

    def backward(self, d_output):
        N, T, input_dim = self.inputs.shape
        n = self.n_units

        for name in self.param_names:
            setattr(self, "d" + name, np.zeros_like(getattr(self, name)))
        self.d_inputs = np.zeros_like(self.inputs)

        # dh_next / dc_next carry dL/dh_t and dL/dc_t backwards across
        # timesteps. dh_next starts as the incoming d_output (gradient at
        # the last timestep); dc_next starts at zero, since nothing after
        # the last timestep uses c_{T-1} directly - only through h_{T-1}.
        dh_next = d_output
        dc_next = np.zeros((N, n))

        for t in reversed(range(T)):
            x_t = self.inputs[:, t, :]
            h_prev = self.h[t]
            c_prev = self.c[t]
            c_t = self.c[t + 1]

            f_t = self.f[t]
            i_t = self.i[t]
            o_t = self.o[t]
            g_t = self.g[t]

            tanh_c_t = np.tanh(c_t)

            # h_t = o_t * tanh(c_t)
            do_t = dh_next * tanh_c_t
            dc_t = dh_next * o_t * (1 - tanh_c_t ** 2) + dc_next

            # c_t = f_t*c_{t-1} + i_t*g_t
            df_t = dc_t * c_prev
            di_t = dc_t * g_t
            dg_t = dc_t * i_t
            dc_prev = dc_t * f_t  # gradient straight into c_{t-1}

            # raw pre-activation gradients: multiply through each gate's
            # own nonlinearity derivative (sigmoid' for f/i/o, tanh' for g)
            dz_f = df_t * f_t * (1 - f_t)
            dz_i = di_t * i_t * (1 - i_t)
            dz_o = do_t * o_t * (1 - o_t)
            dz_g = dg_t * (1 - g_t ** 2)

            self.dWxf += np.dot(x_t.T, dz_f)
            self.dWhf += np.dot(h_prev.T, dz_f)
            self.dbf += np.sum(dz_f, axis=0, keepdims=True)

            self.dWxi += np.dot(x_t.T, dz_i)
            self.dWhi += np.dot(h_prev.T, dz_i)
            self.dbi += np.sum(dz_i, axis=0, keepdims=True)

            self.dWxo += np.dot(x_t.T, dz_o)
            self.dWho += np.dot(h_prev.T, dz_o)
            self.dbo += np.sum(dz_o, axis=0, keepdims=True)

            self.dWxg += np.dot(x_t.T, dz_g)
            self.dWhg += np.dot(h_prev.T, dz_g)
            self.dbg += np.sum(dz_g, axis=0, keepdims=True)

            self.d_inputs[:, t, :] = (
                np.dot(dz_f, self.Wxf.T)
                + np.dot(dz_i, self.Wxi.T)
                + np.dot(dz_o, self.Wxo.T)
                + np.dot(dz_g, self.Wxg.T)
            )

            dh_next = (
                np.dot(dz_f, self.Whf.T)
                + np.dot(dz_i, self.Whi.T)
                + np.dot(dz_o, self.Who.T)
                + np.dot(dz_g, self.Whg.T)
            )
            dc_next = dc_prev

        return self.d_inputs


class Layer_GRU:
    """
    GRU cell:

        z_t = sigmoid(x_t @ Wxz + h_{t-1} @ Whz + bz)                # update gate
        r_t = sigmoid(x_t @ Wxr + h_{t-1} @ Whr + br)                # reset gate
        n_t = tanh(x_t @ Wxn + (r_t * h_{t-1}) @ Whn + bn)           # candidate hidden state
        h_t = (1 - z_t) * h_{t-1} + z_t * n_t
        h_{-1} = 0
        output = h_{T-1}

    GRU has no separate cell state - it merges LSTM's forget/input gates
    into a single update gate z_t (z_t near 1 means "mostly use the new
    candidate", near 0 means "mostly keep the old hidden state"), and the
    reset gate r_t controls how much of the previous hidden state is
    allowed to influence the new candidate n_t. Fewer gates than LSTM (2
    instead of 3, plus the candidate), so fewer parameters, and often
    trains just as well in practice.
    """

    param_names = [
        "Wxz", "Whz", "bz",
        "Wxr", "Whr", "br",
        "Wxn", "Whn", "bn",
    ]

    def __init__(self, n_units):
        self.n_units = n_units
        self.initialized = False

    def forward(self, inputs):
        if not self.initialized:
            input_dim = inputs.shape[2]
            n = self.n_units

            self.Wxz = 0.01 * np.random.randn(input_dim, n)
            self.Whz = 0.01 * np.random.randn(n, n)
            self.bz = np.zeros((1, n))

            self.Wxr = 0.01 * np.random.randn(input_dim, n)
            self.Whr = 0.01 * np.random.randn(n, n)
            self.br = np.zeros((1, n))

            self.Wxn = 0.01 * np.random.randn(input_dim, n)
            self.Whn = 0.01 * np.random.randn(n, n)
            self.bn = np.zeros((1, n))

            self.initialized = True

        self.inputs = inputs
        N, T, input_dim = inputs.shape
        n = self.n_units

        # cache every timestep's gates/states - BPTT needs all of them.
        # h[0] = h_{-1} = 0 (already zeros)
        self.h = np.zeros((T + 1, N, n))
        self.z = np.zeros((T, N, n))
        self.r = np.zeros((T, N, n))
        self.n_candidate = np.zeros((T, N, n))  # candidate hidden state n_t
        self.rh = np.zeros((T, N, n))           # r_t * h_{t-1}, cached for backward

        for t in range(T):
            x_t = inputs[:, t, :]
            h_prev = self.h[t]

            z_t = sigmoid(np.dot(x_t, self.Wxz) + np.dot(h_prev, self.Whz) + self.bz)
            r_t = sigmoid(np.dot(x_t, self.Wxr) + np.dot(h_prev, self.Whr) + self.br)
            rh_t = r_t * h_prev
            n_t = np.tanh(np.dot(x_t, self.Wxn) + np.dot(rh_t, self.Whn) + self.bn)

            h_t = (1 - z_t) * h_prev + z_t * n_t

            self.z[t] = z_t
            self.r[t] = r_t
            self.rh[t] = rh_t
            self.n_candidate[t] = n_t
            self.h[t + 1] = h_t

        self.output = self.h[T]
        return self.output

    def backward(self, d_output):
        N, T, input_dim = self.inputs.shape

        for name in self.param_names:
            setattr(self, "d" + name, np.zeros_like(getattr(self, name)))
        self.d_inputs = np.zeros_like(self.inputs)

        # dh_next carries dL/dh_t backwards across timesteps, same idea
        # as SimpleRNN/LSTM above.
        dh_next = d_output

        for t in reversed(range(T)):
            x_t = self.inputs[:, t, :]
            h_prev = self.h[t]
            z_t = self.z[t]
            r_t = self.r[t]
            n_t = self.n_candidate[t]
            rh_t = self.rh[t]

            # h_t = (1-z_t)*h_{t-1} + z_t*n_t gives two direct paths back
            # to h_{t-1}: one through the (1-z_t) term, one indirectly
            # through n_t's dependence on r_t*h_{t-1} via Whn (below).
            dz_t = dh_next * (n_t - h_prev)
            dn_t = dh_next * z_t
            dh_prev_direct = dh_next * (1 - z_t)

            dz_raw = dz_t * z_t * (1 - z_t)   # through update gate's sigmoid
            dn_raw = dn_t * (1 - n_t ** 2)     # through candidate's tanh

            # n_t depends on r_t*h_{t-1} via Whn
            drh_t = np.dot(dn_raw, self.Whn.T)
            dr_t = drh_t * h_prev
            dh_prev_via_r = drh_t * r_t  # the second (indirect) path back to h_{t-1}

            dr_raw = dr_t * r_t * (1 - r_t)  # through reset gate's sigmoid

            self.dWxz += np.dot(x_t.T, dz_raw)
            self.dWhz += np.dot(h_prev.T, dz_raw)
            self.dbz += np.sum(dz_raw, axis=0, keepdims=True)

            self.dWxr += np.dot(x_t.T, dr_raw)
            self.dWhr += np.dot(h_prev.T, dr_raw)
            self.dbr += np.sum(dr_raw, axis=0, keepdims=True)

            self.dWxn += np.dot(x_t.T, dn_raw)
            self.dWhn += np.dot(rh_t.T, dn_raw)
            self.dbn += np.sum(dn_raw, axis=0, keepdims=True)

            self.d_inputs[:, t, :] = (
                np.dot(dz_raw, self.Wxz.T)
                + np.dot(dr_raw, self.Wxr.T)
                + np.dot(dn_raw, self.Wxn.T)
            )

            # both direct paths back to h_{t-1}, summed together with the
            # z_t/r_t gate contributions through Whz/Whr - mirrors how
            # LSTM sums multiple gate contributions into its dh_prev.
            dh_next = (
                dh_prev_direct
                + dh_prev_via_r
                + np.dot(dz_raw, self.Whz.T)
                + np.dot(dr_raw, self.Whr.T)
            )

        return self.d_inputs
