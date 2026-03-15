# LSTM From Scratch (NumPy)

This project demonstrates a **Long Short-Term Memory (LSTM) neural network implemented completely from scratch using NumPy**.

The goal of this repository is educational: to understand **how LSTMs work internally**, including the forward pass, memory updates, and **Backpropagation Through Time (BPTT)** — without relying on deep learning libraries such as TensorFlow or PyTorch.

---

# What is an LSTM?

LSTM (Long Short-Term Memory) is a type of **Recurrent Neural Network (RNN)** designed to learn patterns in **sequential data** such as:

* text
* speech
* time series
* signals

Traditional RNNs struggle with long sequences due to the **vanishing gradient problem**. LSTM solves this by introducing a **memory cell** that allows information to flow across many time steps.

This memory is controlled by special components called **gates**.

---

# How LSTM Works (Simplified)

At every time step, the LSTM processes:

* **xₜ** → current input
* **hₜ₋₁** → previous hidden state
* **cₜ₋₁** → previous cell state (memory)

The network computes four components:

### 1. Forget Gate

Controls what information from the previous memory should be **removed**.

### 2. Input Gate

Controls what new information should be **stored** in memory.

### 3. Cell Candidate

Creates **new candidate information** that may be added to the memory.

### 4. Output Gate

Controls what information from memory becomes the **output**.

The memory update is:

```
c_t = f_t * c_(t-1) + i_t * g_t
```

The hidden state is:

```
h_t = o_t * tanh(c_t)
```

In this educational implementation, **tanh is used for all gates** to keep the math simple.

---

# What This Project Does

The project trains an LSTM to perform **next-character prediction** on a short text sequence.

Example:

```
Input text: dogs
```

The model learns the pattern:

```
d → o
o → g
g → s
```

After training, the LSTM attempts to predict the **next character** at each step.

---

# Project Structure

```
LSTM/
│
├── main.py
└── train.py
```

### train.py

Contains the full **LSTM implementation**, including:

* weight initialization
* forward pass
* Backpropagation Through Time (BPTT)
* gradient accumulation across timesteps
* SGD weight updates

### main.py

Handles the **training pipeline**:

1. Build vocabulary from input text
2. Encode characters as normalized numbers
3. Train the LSTM using **BPTT**
4. Predict the next character at each timestep

---

# Training Process

For each epoch the model performs:

### 1. Forward Pass

Run the LSTM through all characters in the sequence.

### 2. Loss Computation

Mean Squared Error between predictions and targets.

### 3. Backpropagation Through Time

The sequence is processed **in reverse order** to compute gradients.

### 4. Weight Update

Weights are updated using **Stochastic Gradient Descent (SGD)**.

---

# How to Run the Project

Clone the repository:

```
git clone https://github.com/OmarAhmBelt113/Neural-Networks-from-scratch
```

Navigate to the LSTM folder:

```
cd LSTM
```

Run the program:

```
python main.py
```

Enter a text sequence:

```
Enter text: dogs
```

---

# Example Output

```
Full BPTT Training
Text: 'dogs'

Epoch 1/200  | Loss: 0.256844
Epoch 200/200 | Loss: 0.172910
```

Prediction phase:

```
Step   Char   Input   h_next   c_next   Predicted next
0      'd'   -1.000   0.18952  0.22181  → 'o'
1      'o'    0.333  -0.48189 -0.56020  → 'g'
2      'g'   -0.333   0.31082  0.44959  → 'o'
3      's'    1.000  -0.68394 -0.87769  → 'd'
```

---

# Why This Project Exists

This implementation was created to:

* Understand **how LSTM memory works**
* Learn **Backpropagation Through Time**
* Practice implementing neural networks using **only NumPy**
* Build intuition about **sequence models**

It is designed for **learning and experimentation**, not production use.

---

# Author

**Omar Ahmed Al-Beltagy**
AI Student | Machine Learning Enthusiast
