"""
LSTM Autoencoder for ForexGuard
Advanced deep learning model for sequence-based anomaly detection.

WHY LSTM AUTOENCODER:
1. Captures temporal patterns - detects anomalies in behavioral sequences
2. Learns complex non-linear relationships in user behavior
3. Reconstruction error provides interpretable anomaly score
4. Can detect subtle pattern deviations that tree-based methods miss
5. Works well for sequential forex trading data
6. Self-supervised - no labels needed for training
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
from pathlib import Path
from loguru import logger


class LSTMEncoder(nn.Module):
    """LSTM-based encoder network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last hidden state
        last_hidden = hidden[-1]  # (batch, hidden_dim)

        # Project to latent space
        latent = self.fc(last_hidden)  # (batch, latent_dim)

        return latent


class LSTMDecoder(nn.Module):
    """LSTM-based decoder network."""

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 10,
        seq_len: int = 10,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(latent_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent):
        # Expand latent to sequence
        batch_size = latent.size(0)

        # Project from latent space
        hidden_init = self.fc(latent)  # (batch, hidden_dim)

        # Repeat for each timestep
        decoder_input = hidden_init.unsqueeze(1).repeat(1, self.seq_len, 1)

        # Decode sequence
        lstm_out, _ = self.lstm(decoder_input)  # (batch, seq_len, hidden_dim)

        # Output projection
        output = self.output(lstm_out)  # (batch, seq_len, output_dim)

        return output


class LSTMAutoencoder(nn.Module):
    """Full LSTM Autoencoder for anomaly detection."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 10,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len

        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = LSTMDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def get_reconstruction_error(self, x):
        """Compute per-sample reconstruction error."""
        with torch.no_grad():
            reconstruction = self.forward(x)
            # MSE per sample (mean over seq_len and features)
            error = ((x - reconstruction) ** 2).mean(dim=(1, 2))
        return error


class LSTMAutoencoderDetector:
    """
    Wrapper class for LSTM Autoencoder based anomaly detection.
    Handles training, inference, and threshold-based anomaly detection.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 10,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        threshold_percentile: float = 95.0,
        device: Optional[str] = None
    ):
        """
        Initialize the detector.

        Args:
            input_dim: Number of input features
            seq_len: Sequence length for LSTM
            hidden_dim: Hidden dimension of LSTM
            latent_dim: Latent space dimension
            threshold_percentile: Percentile for anomaly threshold (95 = top 5% are anomalies)
            device: Device to use ('cuda' or 'cpu')
        """
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.threshold_percentile = threshold_percentile

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = LSTMAutoencoder(
            input_dim=input_dim,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        ).to(self.device)

        self.threshold: Optional[float] = None
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_fitted = False
        self.feature_names: list[str] = []

    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create overlapping sequences from feature matrix."""
        sequences = []
        for i in range(len(X) - self.seq_len + 1):
            sequences.append(X[i:i + self.seq_len])
        return np.array(sequences)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize features."""
        if self.mean is None:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
            self.std[self.std == 0] = 1  # Avoid division by zero

        return (X - self.mean) / self.std

    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        validation_split: float = 0.1
    ):
        """
        Train the autoencoder.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Names of features
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Fraction for validation
        """
        logger.info(f"Training LSTM Autoencoder on {len(X)} samples")

        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Handle NaN
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Normalize
        X_norm = self._normalize(X)

        # Create sequences
        sequences = self._create_sequences(X_norm)
        logger.info(f"Created {len(sequences)} sequences of length {self.seq_len}")

        # Split train/val
        n_val = int(len(sequences) * validation_split)
        indices = np.random.permutation(len(sequences))
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        train_data = torch.FloatTensor(sequences[train_idx]).to(self.device)
        val_data = torch.FloatTensor(sequences[val_idx]).to(self.device)

        train_loader = DataLoader(
            TensorDataset(train_data),
            batch_size=batch_size,
            shuffle=True
        )

        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            train_loss = 0
            for batch in train_loader:
                x = batch[0]

                optimizer.zero_grad()
                reconstruction = self.model(x)
                loss = criterion(reconstruction, x)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_reconstruction = self.model(val_data)
                val_loss = criterion(val_reconstruction, val_data).item()
            self.model.train()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Compute threshold from training data
        self.model.eval()
        with torch.no_grad():
            train_errors = self.model.get_reconstruction_error(train_data).cpu().numpy()
            self.threshold = np.percentile(train_errors, self.threshold_percentile)

        logger.info(f"Training complete. Anomaly threshold: {self.threshold:.6f}")
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.

        Args:
            X: Feature matrix (should be sequential data)

        Returns:
            Array of labels: 0 for normal, 1 for anomaly
        """
        scores = self.score_samples(X)
        return (scores > self.threshold).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get reconstruction error scores.

        Args:
            X: Feature matrix

        Returns:
            Array of reconstruction errors (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_norm = (X - self.mean) / self.std

        # Create sequences
        sequences = self._create_sequences(X_norm)

        if len(sequences) == 0:
            return np.array([0.0])

        # Get reconstruction errors
        self.model.eval()
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequences).to(self.device)
            errors = self.model.get_reconstruction_error(seq_tensor).cpu().numpy()

        # Pad to match original length
        padded_errors = np.zeros(len(X))
        padded_errors[:len(errors)] = errors

        return padded_errors

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly probability scores (0 to 1).

        Args:
            X: Feature matrix

        Returns:
            Array of probabilities where higher = more likely anomaly
        """
        scores = self.score_samples(X)

        # Normalize to [0, 1] using threshold as reference
        # Scores above threshold map to > 0.5
        proba = 1 / (1 + np.exp(-(scores - self.threshold) / (self.threshold + 1e-8)))

        return proba

    def get_feature_contributions(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """
        Get per-feature reconstruction errors (contribution to anomaly).

        Args:
            X: Feature matrix

        Returns:
            Dictionary mapping feature names to contribution arrays
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_norm = (X - self.mean) / self.std

        sequences = self._create_sequences(X_norm)

        self.model.eval()
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequences).to(self.device)
            reconstructions = self.model(seq_tensor).cpu().numpy()

        # Per-feature error (averaged over sequence)
        feature_errors = np.abs(sequences - reconstructions).mean(axis=1)  # (n_seq, n_features)

        contributions = {}
        for i, name in enumerate(self.feature_names):
            contributions[name] = feature_errors[:, i]

        return contributions

    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state": self.model.state_dict(),
            "threshold": self.threshold,
            "mean": self.mean,
            "std": self.std,
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }

        torch.save(state, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "LSTMAutoencoderDetector":
        """Load model from disk."""
        state = torch.load(path, map_location="cpu", weights_only=False)

        detector = cls(
            input_dim=state["input_dim"],
            seq_len=state["seq_len"],
            hidden_dim=state["hidden_dim"],
            latent_dim=state["latent_dim"],
            device=device
        )

        detector.model.load_state_dict(state["model_state"])
        detector.threshold = state["threshold"]
        detector.mean = state["mean"]
        detector.std = state["std"]
        detector.feature_names = state["feature_names"]
        detector.is_fitted = state["is_fitted"]

        detector.model.to(detector.device)
        logger.info(f"Model loaded from {path}")

        return detector


if __name__ == "__main__":
    # Test with synthetic sequential data
    np.random.seed(42)

    # Generate normal sequences
    n_samples = 1000
    n_features = 10
    seq_len = 10

    X_normal = np.random.randn(n_samples, n_features)

    # Add some anomalies at the end
    n_anomaly = 50
    X_anomaly = np.random.randn(n_anomaly, n_features) * 3 + 2

    X = np.vstack([X_normal, X_anomaly])

    # Train
    detector = LSTMAutoencoderDetector(
        input_dim=n_features,
        seq_len=seq_len
    )

    feature_names = [f"feature_{i}" for i in range(n_features)]
    detector.fit(X[:n_samples], feature_names=feature_names, epochs=30)

    # Test
    scores = detector.score_samples(X)
    predictions = detector.predict(X)

    print(f"Anomaly threshold: {detector.threshold:.6f}")
    print(f"Normal mean score: {scores[:n_samples-seq_len].mean():.6f}")
    print(f"Anomaly mean score: {scores[n_samples:].mean():.6f}")
    print(f"Detected anomalies: {predictions.sum()}")
