"""PyTorch MLP model wrapped in the BaseModel interface."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, TensorDataset

from src.data.preprocess import build_preprocessing_pipeline
from src.models.base import BaseModel


class _MLP(nn.Module):
    """Simple feed-forward MLP for binary classification (outputs a single logit)."""

    def __init__(
        self, input_dim: int, hidden_dims: list[int], dropout: float
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass returning an un-squeezed logit per sample."""
        return self.net(x).squeeze(-1)  # type: ignore[no-any-return]


class TorchModel(BaseModel):
    """MLP binary classifier built with PyTorch, wrapped in the BaseModel interface.

    Preprocessing (imputation + scaling / one-hot encoding) is handled internally
    via the same :func:`src.data.preprocess.build_preprocessing_pipeline` used by
    the sklearn baselines, ensuring a fair comparison.
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 42,
    ) -> None:
        self._hidden_dims: list[int] = hidden_dims if hidden_dims is not None else [64, 32]
        self._dropout = dropout
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size
        self._random_state = random_state
        self._preprocessor: ColumnTransformer | None = None
        self._net: _MLP | None = None
        self._input_dim: int | None = None
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TorchModel":
        """Preprocess *X*, then train the MLP with BCEWithLogitsLoss.

        Parameters
        ----------
        X:
            Raw feature DataFrame (same schema as the UCI dataset).
        y:
            Binary labels (0 = no disease, 1 = disease).

        Returns
        -------
        TorchModel
            Self, for method chaining.
        """
        torch.manual_seed(self._random_state)

        self._preprocessor = build_preprocessing_pipeline()
        X_arr: np.ndarray = self._preprocessor.fit_transform(X)

        self._input_dim = int(X_arr.shape[1])
        self._net = _MLP(self._input_dim, self._hidden_dims, self._dropout).to(
            self._device
        )

        X_t = torch.tensor(X_arr, dtype=torch.float32).to(self._device)
        y_t = torch.tensor(y.to_numpy(), dtype=torch.float32).to(self._device)

        optimizer = torch.optim.Adam(  # type: ignore[attr-defined]
            self._net.parameters(), lr=self._lr
        )
        criterion = nn.BCEWithLogitsLoss()
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self._batch_size,
            shuffle=True,
        )

        self._net.train()
        for _ in range(self._epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss: torch.Tensor = criterion(self._net(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        self._net.eval()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary class predictions (threshold = 0.5).

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        np.ndarray
            Integer array of shape (n_samples,).
        """
        net = self._require_fitted()
        net.eval()
        with torch.no_grad():
            logits = net(self._to_tensor(X))
            return (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        np.ndarray
            Float array of shape (n_samples, 2) — columns are [P(0), P(1)].
        """
        net = self._require_fitted()
        net.eval()
        with torch.no_grad():
            proba_pos: np.ndarray = (
                torch.sigmoid(net(self._to_tensor(X))).cpu().numpy()
            )
        return np.column_stack([1.0 - proba_pos, proba_pos])

    def get_torch_module(self) -> nn.Module:
        """Return the underlying :class:`_MLP` (raises if the model is not fitted)."""
        return self._require_fitted()

    def save(self, path: Path) -> None:
        """Persist the preprocessor and model weights to *path* using pickle.

        Parameters
        ----------
        path:
            Destination file path (parent directories are created as needed).
        """
        net = self._require_fitted()
        if self._preprocessor is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "preprocessor": self._preprocessor,
            "net_state": net.state_dict(),
            "hidden_dims": self._hidden_dims,
            "dropout": self._dropout,
            "input_dim": self._input_dim,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "TorchModel":
        """Load a TorchModel from *path*.

        Parameters
        ----------
        path:
            Path to a file previously written by :meth:`save`.

        Returns
        -------
        TorchModel
            Fully initialised, eval-mode model ready for inference.
        """
        with path.open("rb") as f:
            payload = pickle.load(f)  # noqa: S301
        instance = cls(
            hidden_dims=payload["hidden_dims"],
            dropout=payload["dropout"],
        )
        instance._preprocessor = payload["preprocessor"]
        instance._input_dim = payload["input_dim"]
        net = _MLP(payload["input_dim"], payload["hidden_dims"], payload["dropout"])
        net.load_state_dict(payload["net_state"])
        net.eval()
        instance._net = net.to(instance._device)
        return instance

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_fitted(self) -> _MLP:
        if self._net is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._net

    def _to_tensor(self, X: pd.DataFrame) -> torch.Tensor:
        if self._preprocessor is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        X_arr: np.ndarray = self._preprocessor.transform(X)
        return torch.tensor(X_arr, dtype=torch.float32).to(self._device)
