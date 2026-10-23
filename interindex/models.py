"""Forecasting models sharing one fit/predict interface."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV


class ZeroBaseline:
    """Predict a 0% return every week: the random-walk benchmark."""
    name = "Zero"

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class MeanBaseline:
    """Predict the average training-period return."""
    name = "Mean"

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class Ridge:
    """Regularised linear regression on the flattened feature window."""
    name = "Ridge"

    def fit(self, X, y):
        self.model_ = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(X.reshape(len(X), -1), y)
        return self

    def predict(self, X):
        return self.model_.predict(X.reshape(len(X), -1))


class _RecurrentNet(nn.Module):
    def __init__(self, cell, n_features, hidden, dropout):
        super().__init__()
        self.rnn = cell(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class RecurrentModel:
    """LSTM/GRU over a (weeks, features) window, averaged over several seeds.

    With ~200 training weeks a single network is very noisy, so we train an
    ensemble and use the last 15% of the training period for early stopping.
    """

    def __init__(self, kind="LSTM", hidden=32, dropout=0.2, lr=1e-3, weight_decay=1e-3,
                 epochs=300, patience=30, seeds=(0, 1, 2, 3, 4)):
        self.name = kind
        self.cell = nn.LSTM if kind == "LSTM" else nn.GRU
        self.hidden, self.dropout, self.lr, self.wd = hidden, dropout, lr, weight_decay
        self.epochs, self.patience, self.seeds = epochs, patience, seeds

    def _train_one(self, seed, Xtr, ytr, Xval, yval):
        torch.manual_seed(seed)
        net = _RecurrentNet(self.cell, Xtr.shape[-1], self.hidden, self.dropout)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_fn = nn.MSELoss()
        best, best_state, bad = np.inf, None, 0
        gen = torch.Generator().manual_seed(seed)
        for _ in range(self.epochs):
            net.train()
            for idx in torch.randperm(len(Xtr), generator=gen).split(32):
                opt.zero_grad()
                loss_fn(net(Xtr[idx]), ytr[idx]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val = loss_fn(net(Xval), yval).item()
            if val < best - 1e-6:
                best, bad = val, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        return net.eval()

    def fit(self, X, y):
        torch.set_num_threads(1)
        self.y_mu_, self.y_sd_ = float(np.mean(y)), float(np.std(y))
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor((y - self.y_mu_) / self.y_sd_, dtype=torch.float32)
        cut = int(len(X) * 0.85)
        self.nets_ = [self._train_one(s, X[:cut], y[:cut], X[cut:], y[cut:]) for s in self.seeds]
        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            preds = np.mean([net(X).numpy() for net in self.nets_], axis=0)
        return preds * self.y_sd_ + self.y_mu_


def all_models():
    return [ZeroBaseline(), MeanBaseline(), Ridge(), RecurrentModel("LSTM"), RecurrentModel("GRU")]
