# LSTM From Scratch (NumPy)

This project demonstrates a **simple implementation of a Long Short-Term Memory (LSTM) network built from scratch using NumPy**.

The goal of this project is educational: to understand **how LSTM networks work internally** without using deep learning frameworks such as TensorFlow or PyTorch.

---

# What is an LSTM?

LSTM (Long Short-Term Memory) is a special type of Recurrent Neural Network (RNN) designed to **remember information for long periods of time**.

Traditional RNNs often suffer from the **vanishing gradient problem**, which makes it difficult to learn long-term dependencies in sequences.

LSTM solves this by introducing a **cell state** and several **gates** that control the flow of information.

---

# How LSTM Works (Simplified)

At each time step the LSTM processes an input and updates two states:

* **Hidden state (h)** → the output of the network
* **Cell state (c)** → the internal memory of the network

The behavior of the LSTM is controlled by several gates:

### 1. Forget Gate

Decides **what information from the previous memory should be removed**.

### 2. Input Gate

Decides **what new information should be added to the memory**.

### 3. Update / Candidate

Creates **new candidate information** that may be stored in the cell state.

### 4. Output Gate

Controls **what part of the memory becomes the output**.

Together these gates allow the LSTM to **learn patterns in sequential data such as text, speech, or time-series data**.

---

# Project Overview

This project contains a minimal implementation of an LSTM network using only **NumPy**.

The model processes text **character by character** and produces a simple prediction for the next character based on the hidden state.

The purpose is to **visualize and understand how the LSTM updates its internal memory step by step**.

---

# Project Structure

```
LSTM/
│
├── main.py
├── train.py
```

### `train.py`

Contains the core **LSTM implementation**, including:

* LSTM weight initialization
* Gate calculations
* Forward pass
* Backward pass (for gradient computation)

The implementation shows how the hidden state and cell state evolve during computation.

### `main.py`

Provides a **simple text prediction pipeline**:

1. Builds a character vocabulary from the input text
2. Encodes characters into normalized numeric values
3. Feeds them sequentially into the LSTM
4. Prints the hidden state and predicted next character at each step

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

Then enter any text:

```
Enter text: hello
```

The program will display the step-by-step LSTM computation and predicted characters.

---

# Example Output

```
Step   Char     Input     h_next      c_next   Predicted next
------------------------------------------------------------
0      'h'      -0.600     0.12345    0.32100  → 'e'
1      'e'      -0.200     0.21300    0.41231  → 'l'
2      'l'       0.200     0.31211    0.51234  → 'l'
...
```

---

# Why This Project Exists

This project was built to:

* Understand **LSTM internals**
* Learn how **gates control memory flow**
* Practice implementing neural networks **from scratch using NumPy**

It is intended for **learning and experimentation**, not production use.

---

# Future Improvements

Possible improvements include:

* Implement full **Backpropagation Through Time (BPTT)**
* Train the model on larger text datasets
* Add **visualizations of gate activations**
* Expand to full **sequence generation**

---

# Author

Omar Ahmed Al-Beltagy
AI Student | Machine Learning Enthusiast
