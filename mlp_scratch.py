"""
Task 7.1 — Multi-Layer Perceptron From Scratch (NumPy Only)

Architecture:
Input (2) → Hidden Layer (4 neurons) → Output (1)

Features:
- Forward Propagation
- Backpropagation
- Gradient Descent
- Cross Entropy Loss
- Loss Curve Saving
- Decision Boundary Saving
- Model Saving (.npz)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# CREATE PROJECT FOLDERS
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

#  XOR DATASET
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

np.random.seed(42)

#  NETWORK PARAMETERS
input_size = 2
hidden_size = 4
output_size = 1

learning_rate = 0.1
epochs = 1000

# Weight initialization
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

#  ACTIVATION FUNCTIONS
def sigmoid(z):
    """Sigmoid activation"""
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    """Derivative of sigmoid"""
    return a * (1 - a)


#  LOSS FUNCTION (Binary Cross Entropy)
def compute_loss(y_true, y_pred):
    epsilon = 1e-8  # prevents log(0)
    loss = -np.mean(
        y_true * np.log(y_pred + epsilon)
        + (1 - y_true) * np.log(1 - y_pred + epsilon)
    )
    return loss


loss_history = []

#  TRAINING LOOP
for epoch in range(epochs):

    # ---------- Forward Propagation ----------
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # ---------- Compute Loss ----------
    loss = compute_loss(y, A2)
    loss_history.append(loss)

    # ---------- Backpropagation ----------
    m = X.shape[0]

    # Output layer gradients
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Hidden layer gradients
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # ---------- Gradient Descent Update ----------
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

#  SAVE MODEL PARAMETERS
model_path = os.path.join(MODEL_DIR, "mlp_weights.npz")

np.savez(
    model_path,
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2
)

print(f"\nModel saved to {model_path}")

# 7. FINAL PREDICTIONS
print("\nFinal Predictions:")
print(np.round(A2, 3))


#  SAVE LOSS CURVE
plt.figure()
plt.plot(loss_history)
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")

loss_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
plt.savefig(loss_path)
plt.close()

print(f"Loss curve saved to {loss_path}")


# 9. PREDICTION FUNCTION
def predict(X_new):
    A1 = sigmoid(np.dot(X_new, W1) + b1)
    A2 = sigmoid(np.dot(A1, W2) + b2)
    return A2


#  DECISION BOUNDARY VISUALIZATION
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]
Z = predict(grid)
Z = Z.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z, alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolors="k")
plt.title("Decision Boundary (MLP From Scratch)")

boundary_path = os.path.join(OUTPUT_DIR, "decision_boundary.png")
plt.savefig(boundary_path)
plt.close()

print(f"Decision boundary saved to {boundary_path}")


# OPTIONAL MODEL LOADER
def load_model(path):
    data = np.load(path)
    return data["W1"], data["b1"], data["W2"], data["b2"]