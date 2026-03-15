# import necessary libraries
import numpy as np

# Define the RNN class
class RNN:
    def __init__(self, vocab, hidden_size=10, learning_rate=0.01):
        self.vocab = sorted(set(vocab))
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.vocab)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        np.random.seed(42)
        self.Wxh = np.random.randn(hidden_size, self.vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(self.vocab_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((self.vocab_size, 1))

    # Convert word index to one-hot vector
    def one_hot(self, idx):
        vec = np.zeros((self.vocab_size, 1))
        vec[idx] = 1
        return vec

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=0)
    
    # Forward pass through the RNN
    def forward(self, inputs, h_prev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        for t in range(len(inputs)):
            xs[t] = self.one_hot(inputs[t])
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = self.softmax(ys[t])
        return xs, hs, ys, ps

    # Backward pass through the RNN
    def backward(self, xs, hs, ps, targets):
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])

        # Compute gradients for each time step in reverse order
        for t in reversed(range(len(targets))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh
            dbh += dh_raw
            dWxh += np.dot(dh_raw, xs[t].T)
            if t > 0:
                dWhh += np.dot(dh_raw, hs[t-1].T)
            dh_next = np.dot(self.Whh.T, dh_raw)

        # Clip gradients to prevent exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        return dWxh, dWhh, dWhy, dbh, dby

    # Update parameters using gradients
    def update(self, grads):
        for param, dparam in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], grads):
            param -= self.learning_rate * dparam
