import numpy as np
import matplotlib.pyplot as plt
import os

# Helper functions

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Neural Network for Regression
class SimpleNN:
    def __init__(self, input_size, hidden_size=10, lr=0.001, epochs=500):
        # Initialize weights
        self.lr = lr
        self.epochs = epochs
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        return self.Z2  # Linear output for regression

    def backward(self, X, y, y_pred):
        m = X.shape[0]
        dZ2 = (y_pred - y) / m  # MSE derivative
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def fit(self, X, y, epochs=500):
        y = np.asarray(y).reshape(-1, 1)
        self.losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = mse_loss(y, y_pred)
            self.losses.append(loss)
            self.backward(X, y, y_pred)
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward(X).ravel()
    
    def plot_loss(self, save_dir="results/figures"):
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(8,5))
        plt.plot(self.losses, label='Training Loss', color='blue')
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.title("SimpleNN Training Loss vs Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        path = f"{save_dir}/simple_nn_loss_curve.png"
        plt.savefig(path)
        print(f"SimpleNN Loss curve saved")
