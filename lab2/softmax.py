import torch
import numpy as np


class SoftmaxClassifier:
    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        self.num_classes = num_classes
        self.W = None
        self.initialize()

    def initialize(self):
        # Shape: (input_size + 1, num_classes) — +1 for bias via bias trick
        self.W = torch.randn(self.input_size + 1, self.num_classes) * 0.01
        self.W.requires_grad_(True)

    @staticmethod
    def _log_softmax(scores):
        # Numerically stable: subtract row-wise max before exp
        m = scores.max(dim=1, keepdim=True).values
        shifted = scores - m
        return shifted - torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))

    def _to_tensor(self, x, dtype=torch.float32):
        if torch.is_tensor(x):
            return x.float()
        return torch.tensor(x, dtype=dtype)

    def _add_bias_column(self, X):
        ones = torch.ones(X.shape[0], 1)
        return torch.cat([X, ones], dim=1)

    def predict(self, X):
        X = self._to_tensor(X)
        X_b = self._add_bias_column(X)
        with torch.no_grad():
            scores = X_b @ self.W
        return scores.argmax(dim=1)

    def predict_proba(self, X):
        X = self._to_tensor(X)
        X_b = self._add_bias_column(X)
        with torch.no_grad():
            scores = X_b @ self.W
            log_probs = self._log_softmax(scores)
        return torch.exp(log_probs)

    def fit(self, X_train, y_train, lr=0.01, reg=1e-4, epochs=10, batch_size=128):
        X_train = self._to_tensor(X_train)
        y_train = self._to_tensor(y_train, dtype=torch.long)
        N = X_train.shape[0]
        X_b = self._add_bias_column(X_train)

        for _ in range(epochs):
            perm = torch.randperm(N)
            for ii in range((N - 1) // batch_size + 1):
                idx = perm[ii * batch_size: (ii + 1) * batch_size]
                xb = X_b[idx]
                yb = y_train[idx]

                scores = xb @ self.W
                log_probs = self._log_softmax(scores)

                data_loss = -log_probs[torch.arange(len(yb)), yb].mean()
                reg_loss = reg * (self.W ** 2).sum()
                loss = data_loss + reg_loss

                loss.backward()
                with torch.no_grad():
                    self.W -= lr * self.W.grad
                    self.W.grad.zero_()

    def save(self, path):
        torch.save(self.W.detach(), path)

    def load(self, path):
        self.W = torch.load(path).requires_grad_(True)
