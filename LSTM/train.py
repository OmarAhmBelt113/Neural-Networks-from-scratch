# ========> LSTM using numpy <=======
import numpy as np

class LSTM:
    def __init__(self):
        # Forget gate weights: W_f·x + U_f·h + b_f
        self.wx = np.random.randn()   # W_f (input)
        self.ww = np.random.randn()   # U_f (hidden)
        self.wb = np.random.randn()   # b_f (bias)

        # Update gate weights: W_u·x + U_u·h + b_u
        self.wu = np.random.randn()   # W_u
        self.wv = np.random.randn()   # U_u
        self.wc = np.random.randn()   # b_u

        # Input gate weights: W_i·x + U_i·h + b_i
        self.wi = np.random.randn()   # W_i
        self.wj = np.random.randn()   # U_i
        self.wk = np.random.randn()   # b_i

        # Cell candidate weights: W_g·x + U_g·h + b_g  (computed but not yet wired into c_next)
        self.wl = np.random.randn()   # W_g
        self.wm = np.random.randn()   # U_g
        self.wn = np.random.randn()   # b_g

        # Output gate weights: W_o·x + U_o·h + b_o
        self.wo = np.random.randn()   # W_o
        self.wp = np.random.randn()   # U_o
        self.wq = np.random.randn()   # b_o

    # ------------------------------------------------------------------
    # Gate activations  (all use tanh here — standard uses sigmoid for
    # f, u, i, o and tanh only for the cell candidate g)
    # ------------------------------------------------------------------

    def forget_gate(self, x, h_prev, c_prev):
        return np.tanh(np.dot(x, self.wx) + np.dot(h_prev, self.ww) + self.wb)

    def update_gate(self, x, h_prev, c_prev):
        return np.tanh(np.dot(x, self.wu) + np.dot(h_prev, self.wv) + self.wc)

    def input_gate(self, x, h_prev, c_prev):
        return np.tanh(np.dot(x, self.wi) + np.dot(h_prev, self.wj) + self.wk)

    def cell_state(self, x, h_prev, c_prev):
        # Cell candidate — computed here but not yet used in c_next below.
        # Wire it in when you're ready: c_next = f*c_prev + u*g
        return np.tanh(np.dot(x, self.wl) + np.dot(h_prev, self.wm) + self.wn)

    def output_gate(self, x, h_prev, c_prev):
        return np.tanh(np.dot(x, self.wo) + np.dot(h_prev, self.wp) + self.wq)

    # ------------------------------------------------------------------
    # Forward pass — returns h_next, c_next and caches for backward
    # ------------------------------------------------------------------

    def forward(self, x, h_prev, c_prev):
        f = self.forget_gate(x, h_prev, c_prev)   # forget gate
        u = self.update_gate(x, h_prev, c_prev)   # update / input modulation
        i = self.input_gate(x, h_prev, c_prev)    # input gate
        g = self.cell_state(x, h_prev, c_prev)    # cell candidate (unused in c_next for now)
        o = self.output_gate(x, h_prev, c_prev)   # output gate

        c_next = f * c_prev + u * i               # cell state update
        h_next = o * np.tanh(c_next)              # hidden state

        # Cache everything needed for backward
        self.cache = (x, h_prev, c_prev, f, u, i, g, o, c_next)
        return h_next, c_next

    # ------------------------------------------------------------------
    # Backward pass
    #
    # dh_next : gradient of loss w.r.t. h_next  (flows in from next timestep or loss)
    # dc_next : gradient of loss w.r.t. c_next  (flows in from next timestep)
    #
    # Returns gradients w.r.t. inputs so you can chain timesteps (BPTT):
    #   dx, dh_prev, dc_prev
    # Also stores weight gradients on self: self.d<weight>
    # ------------------------------------------------------------------

    def backward(self, dh_next, dc_next):
        x, h_prev, c_prev, f, u, i, g, o, c_next = self.cache

        # ---- 1. Gradient through h_next = o * tanh(c_next) ----
        tanh_c_next = np.tanh(c_next)

        do      = dh_next * tanh_c_next                    # ∂L/∂o
        dc_next = dc_next + dh_next * o * (1 - tanh_c_next ** 2)  # add upstream dc

        # ---- 2. Gradient through c_next = f*c_prev + u*i ----
        df      = dc_next * c_prev    # ∂L/∂f
        dc_prev = dc_next * f         # ∂L/∂c_prev  (pass to previous timestep)
        du      = dc_next * i         # ∂L/∂u
        di      = dc_next * u         # ∂L/∂i

        # ---- 3. Backprop through tanh activations: d/dx tanh(z) = 1 - tanh²(z) ----
        dz_f = df * (1 - f ** 2)    # pre-activation gradient for forget gate
        dz_u = du * (1 - u ** 2)    # pre-activation gradient for update gate
        dz_i = di * (1 - i ** 2)    # pre-activation gradient for input gate
        dz_o = do * (1 - o ** 2)    # pre-activation gradient for output gate
        # (cell candidate g is not wired into the graph yet, so dz_g = 0)

        # ---- 4. Weight gradients ----
        # Forget gate
        self.dwx = dz_f * x
        self.dww = dz_f * h_prev
        self.dwb = dz_f

        # Update gate
        self.dwu = dz_u * x
        self.dwv = dz_u * h_prev
        self.dwc = dz_u             # bias gradient (reusing weight name wc)

        # Input gate
        self.dwi = dz_i * x
        self.dwj = dz_i * h_prev
        self.dwk = dz_i

        # Output gate
        self.dwo = dz_o * x
        self.dwp = dz_o * h_prev
        self.dwq = dz_o

        # Cell candidate — zero for now since g is not wired in
        self.dwl = 0.0
        self.dwm = 0.0
        self.dwn = 0.0

        # ---- 5. Input / hidden gradients (to pass to previous layer/timestep) ----
        dx     = dz_f * self.wx + dz_u * self.wu + dz_i * self.wi + dz_o * self.wo
        dh_prev = dz_f * self.ww + dz_u * self.wv + dz_i * self.wj + dz_o * self.wp

        return dx, dh_prev, dc_prev