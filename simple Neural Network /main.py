import numpy as np
from neural_network import forward_pass, calculate_error, backpropagate, update_weights

# Inputs and Outputs
inputs = np.array([[0.05, 0.1]])
targets = np.array([[0.01, 0.99]])

# Weights
w1 = 0.15
w2 = 0.20
w3 = 0.25
w4 = 0.30
w5 = 0.40
w6 = 0.45
w7 = 0.50
w8 = 0.55

# Bias
b1 = 0.35
b2 = 0.60

# Learning Rate
lr = 0.5

# Forward Pass
out_h1, out_h2, out_o1, out_o2 = forward_pass(inputs, w1, w2, w3, w4, w5, w6, w7, w8, b1, b2)

# Error Calculation
E1, E2, E_total = calculate_error(targets, out_o1, out_o2)

print(f"Error for O1: {E1}")
print(f"Error for O2: {E2}")
print(f"Total Error: {E_total}", end="\n\n")

# Backpropagation
dE_total_net_h1, dE_total_net_h2, dE_total_net_o1, dE_total_net_o2 = backpropagate(
    out_o1, out_o2, targets, out_h1, out_h2, w5, w6, w7, w8
)

# Weight Update
w1, w2, w3, w4, w5, w6, w7, w8 = update_weights(
    lr, dE_total_net_h1, dE_total_net_h2, dE_total_net_o1, dE_total_net_o2,
    inputs, out_h1, out_h2, w1, w2, w3, w4, w5, w6, w7, w8
)

# Print updated weights
print(f"Updated w1: {w1}")
print(f"Updated w2: {w2}")
print(f"Updated w3: {w3}")
print(f"Updated w4: {w4}")
print(f"Updated w5: {w5}")
print(f"Updated w6: {w6}")
print(f"Updated w7: {w7}")
print(f"Updated w8: {w8}")