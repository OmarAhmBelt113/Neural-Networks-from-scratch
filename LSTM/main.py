# ========> LSTM Text Prediction Pipeline <=======
import numpy as np
from train import LSTM

# ------------------------------------------------------------------
# 1. Text Encoding
#    Character-level encoding:
#      - Build vocabulary from the input text
#      - Map each character to a normalized float in [-1, 1]
# ------------------------------------------------------------------

def build_vocab(text):
    """Returns char->index and index->char mappings."""
    chars = sorted(set(text))
    char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
    idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char

def encode(text, char_to_idx):
    """Encode text into a list of normalized floats in [-1, 1]."""
    vocab_size = len(char_to_idx)
    indices = [char_to_idx[ch] for ch in text]
    normalized = [(idx / (vocab_size - 1)) * 2 - 1 for idx in indices]
    return indices, normalized

def decode_index(value, idx_to_char, vocab_size):
    """Convert a normalized float back to its closest character."""
    idx = int(round((value + 1) / 2 * (vocab_size - 1)))
    idx = max(0, min(vocab_size - 1, idx))
    return idx_to_char[idx]

# ------------------------------------------------------------------
# 2. Training with full Backpropagation Through Time (BPTT)
#
#    Task: next-character prediction (shift targets by 1).
#    Loss: Mean Squared Error over all output timesteps.
#
#    For each epoch:
#      a) Forward  — run all T timesteps, save every cache
#      b) Loss     — MSE between outputs and targets
#      c) Backward — unwind in reverse (BPTT), accumulate grads
#      d) Update   — SGD step on all weights
# ------------------------------------------------------------------

def train(model, text, epochs=200, lr=0.01):
    char_to_idx, idx_to_char = build_vocab(text)
    vocab_size = len(char_to_idx)
    _, normalized = encode(text, char_to_idx)

    # Inputs are chars 0..T-2, targets are chars 1..T-1
    inputs  = normalized[:-1]
    targets = normalized[1:]
    T = len(inputs)

    if T == 0:
        print("Need at least 2 characters to train.")
        return

    print(f"\n{'='*55}")
    print(f" Full BPTT Training")
    print(f" Text: {repr(text)}  |  T={T}  |  vocab={vocab_size}")
    print(f" Epochs={epochs}  LR={lr}")
    print(f"{'='*55}\n")

    for epoch in range(epochs):

        # ---- (a) Forward pass: collect h outputs and per-step caches ----
        h, c = 0.0, 0.0
        hs      = []   # h_t for every t
        caches  = []   # cache_t for every t (needed by backward_step)

        for t in range(T):
            h, c, cache = model.forward(inputs[t], h, c)
            hs.append(h)
            caches.append(cache)

        hs   = np.array(hs)
        tgts = np.array(targets)

        # ---- (b) MSE loss: L = (1/T) * Σ (h_t - y_t)² ----
        loss = np.mean((hs - tgts) ** 2)

        # ---- (c) BPTT: unwind timesteps in reverse ----
        model.zero_grads()          # clear all accumulated gradients
        dh = 0.0                    # gradient flowing back from t+1
        dc = 0.0                    # cell gradient from t+1

        for t in reversed(range(T)):
            # Gradient of loss w.r.t. h_t (from MSE), then add upstream dh
            dh_from_loss = 2.0 * (hs[t] - tgts[t]) / T
            dh = dh + dh_from_loss

            # Backprop through this timestep; get gradients for t-1
            _, dh, dc = model.backward_step(dh, dc, caches[t])

        # ---- (d) SGD weight update ----
        model.update_weights(lr)

        # Print every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>4}/{epochs}  |  Loss: {loss:.6f}")

    print(f"\n✅ Training complete.\n")

# ------------------------------------------------------------------
# 3. Prediction (inference only — no backward)
# ------------------------------------------------------------------

def predict(model, text):
    char_to_idx, idx_to_char = build_vocab(text)
    vocab_size = len(char_to_idx)
    indices, normalized = encode(text, char_to_idx)

    print(f"\n{'='*55}")
    print(f" Prediction")
    print(f" Text: {repr(text)}  |  vocab={vocab_size}")
    print(f"{'='*55}")
    print(f"\n{'Step':<6} {'Char':<6} {'Input':>8} {'h_next':>10} {'c_next':>10}  Predicted next")
    print("-" * 60)

    h, c = 0.0, 0.0
    for t, (ch, x) in enumerate(zip(text, normalized)):
        h, c, _ = model.forward(x, h, c)          # unpack 3 values
        predicted_char = decode_index(h, idx_to_char, vocab_size)
        print(f"{t:<6} {repr(ch):<6} {x:>8.3f} {h:>10.5f} {c:>10.5f}  → {repr(predicted_char)}")

    print(f"\n✅ Done.  Final h = {h:.5f}  |  Final c = {c:.5f}")

# ------------------------------------------------------------------
# 4. Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    model = LSTM()

    text = input("Enter text: ").strip()
    if not text:
        text = "hello"
        print(f"(no input given, using default: {repr(text)})")

    # Train with BPTT, then show predictions
    train(model, text, epochs=200, lr=0.01)
    predict(model, text)
