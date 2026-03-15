# import necessary libraries
from rnn import RNN
import numpy as np

# Input sequence
text = ["I", "am", "Omar", "Al-beltagy"]
rnn = RNN(text, hidden_size=10, learning_rate=0.01)

# Convert text to indices
sequence = [rnn.word_to_idx[word] for word in text]
h_prev = np.zeros((rnn.hidden_size, 1))

# Training loop
for epoch in range(1500):
    inputs = sequence[:3]
    target = [sequence[3]]  # Predict "best" from "barca is the"

    xs, hs, ys, ps = rnn.forward(inputs, h_prev)
    loss = -np.log(ps[len(inputs)-1][target[0], 0])
    grads = rnn.backward(xs, hs, ps, target)
    rnn.update(grads)

    h_prev = hs[len(inputs) - 1]

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Test prediction
xs, hs, ys, ps = rnn.forward(sequence[:3], np.zeros((rnn.hidden_size, 1)))
predicted_idx = np.argmax(ps[2])
predicted_word = rnn.idx_to_word[predicted_idx]
print(f"Predicted word: {predicted_word}")