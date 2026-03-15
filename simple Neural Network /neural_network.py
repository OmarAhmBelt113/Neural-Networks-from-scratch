import numpy as np

# Activation Function (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Forward Pass
def forward_pass(inputs, w1, w2, w3, w4, w5, w6, w7, w8, b1, b2):
    # Hidden Layer
    net_h1 = inputs[0, 0] * w1 + inputs[0, 1] * w3 + b1
    out_h1 = sigmoid(net_h1)

    net_h2 = inputs[0, 0] * w2 + inputs[0, 1] * w4 + b1
    out_h2 = sigmoid(net_h2)

    # Output Layer
    net_o1 = out_h1 * w5 + out_h2 * w7 + b2
    out_o1 = sigmoid(net_o1)

    net_o2 = out_h1 * w6 + out_h2 * w8 + b2
    out_o2 = sigmoid(net_o2)

    return out_h1, out_h2, out_o1, out_o2

# Error Calculation
def calculate_error(targets, out_o1, out_o2):
    E1 = 0.5 * (targets[0, 0] - out_o1) ** 2
    E2 = 0.5 * (targets[0, 1] - out_o2) ** 2
    E_total = E1 + E2
    return E1, E2, E_total

# Backpropagation
def backpropagate(out_o1, out_o2, targets, out_h1, out_h2, w5, w6, w7, w8):
    # Output Layer
    dE_total_out_o1 = out_o1 - targets[0, 0]
    dE_total_net_o1 = dE_total_out_o1 * sigmoid_derivative(out_o1)

    dE_total_out_o2 = out_o2 - targets[0, 1]
    dE_total_net_o2 = dE_total_out_o2 * sigmoid_derivative(out_o2)

    # Hidden Layer
    dE_total_out_h1 = (dE_total_net_o1 * w5) + (dE_total_net_o2 * w6)
    dE_total_net_h1 = dE_total_out_h1 * sigmoid_derivative(out_h1)

    dE_total_out_h2 = (dE_total_net_o1 * w7) + (dE_total_net_o2 * w8)
    dE_total_net_h2 = dE_total_out_h2 * sigmoid_derivative(out_h2)

    return dE_total_net_h1, dE_total_net_h2, dE_total_net_o1, dE_total_net_o2

# Weight Update
def update_weights(lr, dE_total_net_h1, dE_total_net_h2, dE_total_net_o1, dE_total_net_o2,
                   inputs, out_h1, out_h2, w1, w2, w3, w4, w5, w6, w7, w8):
    w1 = w1 - lr * dE_total_net_h1 * inputs[0, 0]
    w2 = w2 - lr * dE_total_net_h2 * inputs[0, 0]
    w3 = w3 - lr * dE_total_net_h1 * inputs[0, 1]
    w4 = w4 - lr * dE_total_net_h2 * inputs[0, 1]
    w5 = w5 - lr * dE_total_net_o1 * out_h1
    w6 = w6 - lr * dE_total_net_o2 * out_h1
    w7 = w7 - lr * dE_total_net_o1 * out_h2
    w8 = w8 - lr * dE_total_net_o2 * out_h2

    return w1, w2, w3, w4, w5, w6, w7, w8