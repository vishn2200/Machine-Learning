import torch
import numpy as np


class GMMModel:
    def __init__(self, n_components):
        self.n_components = n_components
        self.weights = torch.ones(n_components) / n_components
        self.means = torch.randn(n_components, 3)
        self.covariances = torch.zeros(n_components, 3, 3)

    def fit(self, X, max_iters=100, tol=1e-4):
        n_samples = X.shape[0]
        n_features = X.shape[1]

        for iteration in range(max_iters):
            # Expectation step
            responsibilities = self._e_step(X)

            # Maximization step
            self._m_step(X, responsibilities)

            if self._is_converged(X, responsibilities, tol):
                break

    def _e_step(self, X):
        n_samples = X.shape[0]
        n_components = self.n_components
        responsibilities = torch.zeros(n_components, n_samples)

        for i in range(n_components):
            diff = X - self.means[i]
            weighted_exponent = self.weights[i] * torch.exp(
                -0.5 * torch.sum(
                    diff * (torch.matmul(diff, self._inverse(self.covariances[i]))),
                    dim=1,
                )
            )
            responsibilities[i] = weighted_exponent

        responsibilities /= responsibilities.sum(0)
        return responsibilities

    def _m_step(self, X, responsibilities):
        n_samples = X.shape[0]
        total_weight = responsibilities.sum(1)
        self.weights = total_weight / n_samples

        for i in range(self.n_components):
            weighted_X = responsibilities[i].unsqueeze(1) * X
            self.means[i] = torch.sum(weighted_X, dim=0) / total_weight[i]
            diff = X - self.means[i]
            self.covariances[i] = (diff * (responsibilities[i].unsqueeze(1))).transpose(0, 1) @ diff / total_weight[i]


    def _inverse(self, matrix):
        return torch.inverse(matrix + torch.eye(matrix.shape[0]) * 1e-6)

    def _is_converged(self, X, responsibilities, tol):
        prev_log_likelihood = self._log_likelihood(X, responsibilities)
        responsibilities = self._e_step(X)
        current_log_likelihood = self._log_likelihood(X, responsibilities)
        return abs(current_log_likelihood - prev_log_likelihood) < tol

    def _log_likelihood(self, X, responsibilities):
        log_likelihood = torch.log(responsibilities.sum(0)).sum()
        return log_likelihood

    def predict(self, X):
        responsibilities = self._e_step(X)
        labels = torch.argmax(responsibilities, dim=0)
        return labels

    def get_cluster_means(self):
        return self.means

    def get_cluster_covariances(self):
        return self.covariances
