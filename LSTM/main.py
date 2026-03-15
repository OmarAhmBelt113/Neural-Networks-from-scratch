# ========> LSTM Text Prediction Pipeline <=======
import numpy as np
from train import LSTM

# ------------------------------------------------------------------
# 1. Text Encoding
#    We do character-level encoding:
#      - Build a vocabulary from the input text
#      - Map each character to an integer index
#      - Normalize to [-1, 1] so it matches the scalar weight setup
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
    # Normalize: map [0, vocab_size-1] → [-1, 1]
    normalized = [(idx / (vocab_size - 1)) * 2 - 1 for idx in indices]
    return indices, normalized

def decode_index(value, idx_to_char, vocab_size):
    """Convert a normalized float back to its closest character."""
    # Map [-1, 1] → [0, vocab_size-1] and round to nearest index
    idx = int(round((value + 1) / 2 * (vocab_size - 1)))
    idx = max(0, min(vocab_size - 1, idx))   # clamp
    return idx_to_char[idx]

# ------------------------------------------------------------------
# 2. Forward pass over the full text sequence
#    Each character is one timestep fed into the LSTM.
#    After each timestep we record h_next as the "output".
# ------------------------------------------------------------------

def predict(model, text):
    char_to_idx, idx_to_char = build_vocab(text)
    vocab_size = len(char_to_idx)

    indices, normalized = encode(text, char_to_idx)
    print(f"\nText      : {repr(text)}")
    print(f"Vocab     : {''.join(sorted(char_to_idx.keys()))!r}  ({vocab_size} unique chars)")
    print(f"Encoded   : {indices}")
    print(f"Normalized: {[round(v, 3) for v in normalized]}\n")

    # Initial hidden and cell states
    h = 0.0
    c = 0.0

    print(f"{'Step':<6} {'Char':<6} {'Input':>8} {'h_next':>10} {'c_next':>10}  Predicted next")
    print("-" * 60)

    hidden_states = []
    for t, (ch, x) in enumerate(zip(text, normalized)):
        h, c = model.forward(x, h, c)
        hidden_states.append(h)

        # Predict the next character from the current hidden state
        predicted_char = decode_index(h, idx_to_char, vocab_size)

        print(f"{t:<6} {repr(ch):<6} {x:>8.3f} {h:>10.5f} {c:>10.5f}  → {repr(predicted_char)}")

    print("\n✅ Done. Final hidden state (h) = {:.5f}".format(h))
    print("   Final cell state   (c) = {:.5f}".format(c))
    return hidden_states

# ------------------------------------------------------------------
# 3. Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    model = LSTM()

    # You can change this to any text you like
    text = input("Enter text: ").strip()
    if not text:
        text = "hello"
        print(f"(no input given, using default: {repr(text)})")

    predict(model, text)
