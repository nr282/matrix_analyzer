"""
Deep learning implementation with full backpropagation.

Implements a modular neural network framework where each layer stores its
inputs/outputs during the forward pass so gradients can be computed during
the backward pass. Weight matrices at each layer are the primary objects
of interest for the matrix analysis goals of this repository.
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._mask


class Sigmoid:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = 1.0 / (1.0 + np.exp(-x))
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._out * (1.0 - self._out)


class Tanh:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1.0 - self._out ** 2)


class Softmax:
    def forward(self, x: np.ndarray) -> np.ndarray:
        shifted = x - x.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        self._out = exp / exp.sum(axis=1, keepdims=True)
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Jacobian-vector product: dL/dx_i = p_i * (dL/dp_i - sum_j(dL/dp_j * p_j))
        dot = (grad * self._out).sum(axis=1, keepdims=True)
        return self._out * (grad - dot)


# ---------------------------------------------------------------------------
# Linear layer
# ---------------------------------------------------------------------------

class Linear:
    """
    Fully-connected layer: out = X @ W + b

    W has shape (in_features, out_features).
    This is the primary source of backpropagation matrices studied in this repo.
    """

    def __init__(self, in_features: int, out_features: int, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        # He initialisation for layers followed by ReLU
        scale = np.sqrt(2.0 / in_features)
        self.W = rng.standard_normal((in_features, out_features)) * scale
        self.b = np.zeros(out_features)

        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # grad shape: (batch, out_features)
        self.dW = self._x.T @ grad                   # (in_features, out_features)
        self.db = grad.sum(axis=0)                   # (out_features,)
        return grad @ self.W.T                       # (batch, in_features)

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class MSELoss:
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self._diff = pred - target
        return float(np.mean(self._diff ** 2))

    def backward(self) -> np.ndarray:
        return 2.0 * self._diff / self._diff.size


class CrossEntropyLoss:
    """Numerically stable cross-entropy for class-index targets."""

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        self._probs = probs
        self._targets = targets
        n = logits.shape[0]
        log_likelihood = -np.log(probs[np.arange(n), targets] + 1e-12)
        return float(log_likelihood.mean())

    def backward(self) -> np.ndarray:
        n = self._probs.shape[0]
        grad = self._probs.copy()
        grad[np.arange(n), self._targets] -= 1.0
        return grad / n


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

class SGD:
    def __init__(self, lr: float = 1e-2, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self._velocity: dict = {}

    def step(self, layers: list):
        for layer in layers:
            if not isinstance(layer, Linear):
                continue
            for idx, (param, grad) in enumerate(layer.parameters()):
                if grad is None:
                    continue
                key = (id(layer), idx)
                v = self._velocity.get(key, np.zeros_like(param))
                v = self.momentum * v + grad
                self._velocity[key] = v
                param -= self.lr * v


class Adam:
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: dict = {}
        self._v: dict = {}
        self._t = 0

    def step(self, layers: list):
        self._t += 1
        for layer in layers:
            if not isinstance(layer, Linear):
                continue
            for idx, (param, grad) in enumerate(layer.parameters()):
                if grad is None:
                    continue
                key = (id(layer), idx)
                m = self._m.get(key, np.zeros_like(param))
                v = self._v.get(key, np.zeros_like(param))
                m = self.beta1 * m + (1.0 - self.beta1) * grad
                v = self.beta2 * v + (1.0 - self.beta2) * grad ** 2
                self._m[key] = m
                self._v[key] = v
                m_hat = m / (1.0 - self.beta1 ** self._t)
                v_hat = v / (1.0 - self.beta2 ** self._t)
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Sequential model
# ---------------------------------------------------------------------------

class Sequential:
    """
    Container that chains layers and activations for forward/backward passes.

    Example
    -------
    model = Sequential([
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10),
    ])
    """

    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad: np.ndarray) -> np.ndarray:
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def linear_layers(self) -> list:
        return [l for l in self.layers if isinstance(l, Linear)]

    def weight_matrices(self) -> list[np.ndarray]:
        """Return all weight matrices W in forward order — for matrix analysis."""
        return [l.W for l in self.linear_layers()]

    def gradient_matrices(self) -> list[np.ndarray]:
        """Return all weight gradient matrices dW after a backward pass."""
        return [l.dW for l in self.linear_layers() if l.dW is not None]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model: Sequential,
          X: np.ndarray,
          y: np.ndarray,
          loss_fn,
          optimizer,
          epochs: int = 100,
          batch_size: int = 32,
          verbose: bool = True) -> list[float]:
    n = X.shape[0]
    history = []

    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(n)
        epoch_loss = 0.0
        steps = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            xb, yb = X[batch_idx], y[batch_idx]

            pred = model.forward(xb)
            loss = loss_fn.forward(pred, yb)
            grad = loss_fn.backward()
            model.backward(grad)
            optimizer.step(model.layers)

            epoch_loss += loss
            steps += 1

        avg_loss = epoch_loss / steps
        history.append(avg_loss)

        if verbose and (epoch % max(1, epochs // 10) == 0):
            print(f"Epoch {epoch:4d}/{epochs}  loss={avg_loss:.6f}")

    return history


# ---------------------------------------------------------------------------
# Demo — large synthetic binary classification (100 000 obs, 100 features)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    N, F = 100_000, 100
    X = rng.standard_normal((N, F))

    # Ground-truth: label = sign of a sparse linear projection + nonlinearity
    true_w = rng.standard_normal(F)
    true_w[50:] = 0.0          # only first 50 features matter
    y = (X @ true_w + rng.standard_normal(N) * 0.5 > 0).astype(int)

    split = int(0.8 * N)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        Linear(F, 256, seed=42),
        ReLU(),
        Linear(256, 128, seed=43),
        ReLU(),
        Linear(128, 64, seed=44),
        ReLU(),
        Linear(64, 2, seed=45),
    ])

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(lr=1e-3)

    history = train(model, X_train, y_train, loss_fn, optimizer,
                    epochs=20, batch_size=256, verbose=True)

    logits = model.forward(X_test)
    preds = logits.argmax(axis=1)
    print(f"\nTest accuracy: {(preds == y_test).mean():.2%}  "
          f"({N - split} held-out samples)")

    print("\nWeight matrix shapes (for matrix analysis):")
    for i, W in enumerate(model.weight_matrices()):
        print(f"  Layer {i}: W.shape={W.shape}")

    print("\nGradient matrix shapes (after last backward pass):")
    for i, dW in enumerate(model.gradient_matrices()):
        print(f"  Layer {i}: dW.shape={dW.shape}")
