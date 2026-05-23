# Micrograd From Scratch

A tiny neural network and automatic differentiation engine inspired by micrograd by Andrej Karpathy.

This project implements:

- Reverse-mode automatic differentiation
- Computational graph construction
- Backpropagation
- Neurons
- Layers
- Multi-layer perceptrons (MLP)

The goal is to understand how deep learning frameworks like PyTorch work internally.

---

# Project Structure

```text
project/
│
├── micrograd/
│   ├── __init__.py
│   ├── engine.py
│   └── nn.py
│
└── main.py
```

---

# Features

## Automatic Differentiation Engine

Implemented in `engine.py`

Supports:

- Addition
- Multiplication
- Subtraction
- Division
- Power
- Tanh activation
- Exponential function
- Backpropagation using topological sorting

---

## Neural Network Components

Implemented in `nn.py`

Includes:

- `Neuron`
- `Layer`
- `MLP` (Multi-Layer Perceptron)

---

# How It Works

Each mathematical operation creates a new `Value` object.

Example:

```python
a = Value(2.0)
b = Value(3.0)

c = a * b
d = c + 4
```

This builds a computational graph internally.

Calling:

```python
d.backward()
```

computes gradients for all dependent nodes using reverse-mode autodiff.

---

# Computational Graph Example

```text
a ----\
       (*) ---- c ---- (+) ---- d
b ----/                    ^
                            |
                            4
```

Backpropagation traverses this graph in reverse topological order.

---

# Example Usage

## Create an MLP

```python
from micrograd.nn import MLP

n = MLP(3, [4, 4, 1])
```

Architecture:

```text
3 inputs
→ 4 neurons
→ 4 neurons
→ 1 output
```

---

## Forward Pass

```python
x = [2.0, 3.0, -1.0]

output = n(x)

print(output)
```

---

## Backward Pass

```python
output.backward()
```

After this:

```python
for p in n.parameters():
    print(p.grad)
```

will print gradients for all parameters.

---

# Training Example

```python
from micrograd.nn import MLP

model = MLP(3, [4, 4, 1])

x = [2.0, 3.0, -1.0]

y = model(x)

loss = (y - 1.0) ** 2

model.zero_grad()
loss.backward()

learning_rate = 0.05

for p in model.parameters():
    p.data += -learning_rate * p.grad

print(loss.data)
```

---

# Core Concepts Learned

This project demonstrates:

- Computational graphs
- Chain rule
- Reverse-mode autodiff
- Gradient accumulation
- Backpropagation
- Neural network architecture
- Parameter optimization

---

# Important Classes

## Value

Represents:

- Scalar value
- Gradient
- Graph connections
- Backward function

Example:

```python
x = Value(2.0)
```

---

## Neuron

Implements:

```text
w*x + b
```

followed by activation:

```text
tanh()
```

---

## Layer

Collection of neurons.

---

## MLP

Stack of layers.

---

# Run The Project

## Clone Repository

```bash
git clone <your-repo-url>
```

---

## Run Example

```bash
python main.py
```

---

# Future Improvements

Possible additions:

- ReLU activation
- Softmax
- Cross-entropy loss
- SGD optimizer
- Adam optimizer
- Matrix/tensor support
- GPU acceleration
- Visualization of computation graph

---

# Inspiration

Inspired by:

- https://github.com/karpathy/micrograd
- Andrej Karpathy
- PyTorch internals

---

# License

MIT License
