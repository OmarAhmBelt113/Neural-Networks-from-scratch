# ========> LSTM using numpy <=======
import numpy as np

class LSTM:
    def __init__(self):
        # Forget gate weights: W_f·x + U_f·h + b_f
        self.wx = np.random.randn()   # W_f (input)
        self.ww = np.random.randn()   # U_f (hidden)
        self.wb = np.random.randn()   # b_f (bias)

        # Input gate weights: W_i·x + U_i·h + b_i
        self.wi = np.random.randn()   # W_i
        self.wj = np.random.randn()   # U_i
        self.wk = np.random.randn()   # b_i

        # Cell candidate weights: W_g·x + U_g·h + b_g
        self.wl = np.random.randn()   # W_g
        self.wm = np.random.randn()   # U_g
        self.wn = np.random.randn()   # b_g

        # Output gate weights: W_o·x + U_o·h + b_o
        self.wo = np.random.randn()   # W_o
        self.wp = np.random.randn()   # U_o
        self.wq = np.random.randn()   # b_o

    # ------------------------------------------------------------------
    # Gate activations
    # Standard LSTM uses sigmoid for f, i, o and tanh for g.
    # We use tanh everywhere for simplicity (educational implementation).
    # ------------------------------------------------------------------

    def _forget_gate(self, x, h_prev):
        return np.tanh(x * self.wx + h_prev * self.ww + self.wb)

    def _input_gate(self, x, h_prev):
        return np.tanh(x * self.wi + h_prev * self.wj + self.wk)

    def _cell_candidate(self, x, h_prev):
        return np.tanh(x * self.wl + h_prev * self.wm + self.wn)

    def _output_gate(self, x, h_prev):
        return np.tanh(x * self.wo + h_prev * self.wp + self.wq)

    # ------------------------------------------------------------------
    # Forward pass
    #
    # Returns h_next, c_next, and a cache tuple needed for backward.
    # Each call saves its own cache so every timestep can be unwound
    # independently during BPTT.
    # ------------------------------------------------------------------

    def forward(self, x, h_prev, c_prev):
        f = self._forget_gate(x, h_prev)      # forget gate
        i = self._input_gate(x, h_prev)       # input gate
        g = self._cell_candidate(x, h_prev)   # cell candidate
        o = self._output_gate(x, h_prev)      # output gate

        c_next = f * c_prev + i * g           # standard LSTM cell state
        h_next = o * np.tanh(c_next)          # hidden state

        cache = (x, h_prev, c_prev, f, i, g, o, c_next)
        return h_next, c_next, cache

    # ------------------------------------------------------------------
    # Gradient bookkeeping helpers
    # ------------------------------------------------------------------

    def zero_grads(self):
        """Reset all accumulated weight gradients to zero before a new pass."""
        self.dwx = self.dww = self.dwb = 0.0
        self.dwi = self.dwj = self.dwk = 0.0
        self.dwl = self.dwm = self.dwn = 0.0
        self.dwo = self.dwp = self.dwq = 0.0

    # ------------------------------------------------------------------
    # Backward pass — single timestep
    #
    # dh_next : ∂L/∂h_t  (from loss at t  PLUS  gradient from t+1)
    # dc_next : ∂L/∂c_t  (flowing back from t+1 through the cell highway)
    # cache   : saved activations for this specific timestep
    #
    # Uses += so gradients ACCUMULATE across all T timesteps (true BPTT).
    #
    # Returns (dx, dh_prev, dc_prev) to chain into timestep t-1.
    # ------------------------------------------------------------------

    def backward_step(self, dh_next, dc_next, cache):
        x, h_prev, c_prev, f, i, g, o, c_next = cache

        # ---- 1. Gradient through h_next = o * tanh(c_next) ----
        tanh_c = np.tanh(c_next)
        do      = dh_next * tanh_c
        dc_next = dc_next + dh_next * o * (1 - tanh_c ** 2)  # add upstream dc

        # ---- 2. Gradient through c_next = f*c_prev + i*g ----
        df      = dc_next * c_prev   # ∂L/∂f
        dc_prev = dc_next * f        # ∂L/∂c_prev  → previous timestep
        di      = dc_next * g        # ∂L/∂i
        dg      = dc_next * i        # ∂L/∂g

        # ---- 3. Backprop through tanh: d/dz tanh(z) = 1 - tanh²(z) ----
        dz_f = df * (1 - f ** 2)
        dz_i = di * (1 - i ** 2)
        dz_g = dg * (1 - g ** 2)
        dz_o = do * (1 - o ** 2)

        # ---- 4. Accumulate weight gradients across timesteps (BPTT) ----
        self.dwx += dz_f * x;  self.dww += dz_f * h_prev;  self.dwb += dz_f
        self.dwi += dz_i * x;  self.dwj += dz_i * h_prev;  self.dwk += dz_i
        self.dwl += dz_g * x;  self.dwm += dz_g * h_prev;  self.dwn += dz_g
        self.dwo += dz_o * x;  self.dwp += dz_o * h_prev;  self.dwq += dz_o

        # ---- 5. Pass gradients to previous timestep ----
        dx      = dz_f * self.wx + dz_i * self.wi + dz_g * self.wl + dz_o * self.wo
        dh_prev = dz_f * self.ww + dz_i * self.wj + dz_g * self.wm + dz_o * self.wp

        return dx, dh_prev, dc_prev

    # ------------------------------------------------------------------
    # SGD weight update
    # ------------------------------------------------------------------

    def update_weights(self, lr):
        self.wx -= lr * self.dwx;  self.ww -= lr * self.dww;  self.wb -= lr * self.dwb
        self.wi -= lr * self.dwi;  self.wj -= lr * self.dwj;  self.wk -= lr * self.dwk
        self.wl -= lr * self.dwl;  self.wm -= lr * self.dwm;  self.wn -= lr * self.dwn
        self.wo -= lr * self.dwo;  self.wp -= lr * self.dwp;  self.wq -= lr * self.dwq