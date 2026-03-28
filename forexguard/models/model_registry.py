"""
Model Registry for ForexGuard
Handles model versioning, loading, and management.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
from loguru import logger

from .isolation_forest import IsolationForestDetector
from .lstm_autoencoder import LSTMAutoencoderDetector


MODEL_TYPES = {
    "isolation_forest": IsolationForestDetector,
    "lstm_autoencoder": LSTMAutoencoderDetector
}


class ModelRegistry:
    """
    Registry for managing trained anomaly detection models.
    Supports multiple model types and versioning.
    """

    def __init__(self, base_path: str = "models/saved"):
        """
        Initialize the registry.

        Args:
            base_path: Base directory for storing models
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.base_path / "registry.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load registry metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {"models": {}, "active": {}}

    def _save_metadata(self):
        """Save registry metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def register(
        self,
        model: Union[IsolationForestDetector, LSTMAutoencoderDetector],
        model_name: str,
        model_type: str,
        description: str = "",
        metrics: Optional[dict] = None,
        set_active: bool = True
    ) -> str:
        """
        Register and save a trained model.

        Args:
            model: Trained model instance
            model_name: Name for the model
            model_type: Type of model ('isolation_forest' or 'lstm_autoencoder')
            description: Optional description
            metrics: Optional training/validation metrics
            set_active: Whether to set this as the active model for its type

        Returns:
            Version string for the registered model
        """
        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unknown model type: {model_type}. Must be one of {list(MODEL_TYPES.keys())}")

        # Generate version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v_{timestamp}"

        # Create model directory
        model_dir = self.base_path / model_type / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Determine file extension
        if model_type == "isolation_forest":
            model_path = model_dir / f"{version}.joblib"
        else:
            model_path = model_dir / f"{version}.pt"

        # Save model
        model.save(str(model_path))

        # Update metadata
        model_key = f"{model_type}/{model_name}"
        if model_key not in self.metadata["models"]:
            self.metadata["models"][model_key] = {"versions": []}

        self.metadata["models"][model_key]["versions"].append({
            "version": version,
            "path": str(model_path),
            "created_at": datetime.now().isoformat(),
            "description": description,
            "metrics": metrics or {}
        })

        if set_active:
            self.metadata["active"][model_type] = {
                "name": model_name,
                "version": version,
                "path": str(model_path)
            }

        self._save_metadata()

        logger.info(f"Registered model {model_name} ({model_type}) version {version}")
        return version

    def load(
        self,
        model_type: str,
        model_name: Optional[str] = None,
        version: Optional[str] = None
    ) -> Union[IsolationForestDetector, LSTMAutoencoderDetector]:
        """
        Load a model from the registry.

        Args:
            model_type: Type of model to load
            model_name: Name of model (if None, loads active model)
            version: Specific version (if None, loads latest)

        Returns:
            Loaded model instance
        """
        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unknown model type: {model_type}")

        # Determine which model to load
        if model_name is None:
            # Load active model
            if model_type not in self.metadata["active"]:
                raise ValueError(f"No active model for type: {model_type}")
            active = self.metadata["active"][model_type]
            model_path = active["path"]
        else:
            model_key = f"{model_type}/{model_name}"
            if model_key not in self.metadata["models"]:
                raise ValueError(f"Model not found: {model_key}")

            versions = self.metadata["models"][model_key]["versions"]

            if version:
                # Find specific version
                matching = [v for v in versions if v["version"] == version]
                if not matching:
                    raise ValueError(f"Version {version} not found for {model_key}")
                model_path = matching[0]["path"]
            else:
                # Load latest version
                model_path = versions[-1]["path"]

        # Load model
        model_class = MODEL_TYPES[model_type]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        return model_class.load(model_path)

    def list_models(self, model_type: Optional[str] = None) -> list[dict]:
        """List all registered models."""
        models = []

        for key, data in self.metadata["models"].items():
            m_type, m_name = key.split("/")

            if model_type and m_type != model_type:
                continue

            latest = data["versions"][-1] if data["versions"] else None
            is_active = (
                self.metadata["active"].get(m_type, {}).get("name") == m_name
            )

            models.append({
                "type": m_type,
                "name": m_name,
                "versions": len(data["versions"]),
                "latest_version": latest["version"] if latest else None,
                "created_at": latest["created_at"] if latest else None,
                "is_active": is_active
            })

        return models

    def get_active_model(self, model_type: str) -> Optional[dict]:
        """Get info about the active model for a type."""
        return self.metadata["active"].get(model_type)

    def get_active_model_id(self) -> Optional[str]:
        """Get the ID of the currently active model (any type)."""
        # Return the first active model found
        for model_type, info in self.metadata["active"].items():
            return f"{model_type}_{info['name']}_{info['version']}"
        return None

    def load_active(self) -> Union[IsolationForestDetector, LSTMAutoencoderDetector]:
        """Load the currently active model (any type)."""
        for model_type in ["isolation_forest", "lstm_autoencoder"]:
            if model_type in self.metadata["active"]:
                return self.load(model_type)
        raise ValueError("No active model found. Train a model first.")

    def set_active(self, model_type: str, model_name: str, version: str):
        """Set the active model for a type."""
        model_key = f"{model_type}/{model_name}"

        if model_key not in self.metadata["models"]:
            raise ValueError(f"Model not found: {model_key}")

        versions = self.metadata["models"][model_key]["versions"]
        matching = [v for v in versions if v["version"] == version]

        if not matching:
            raise ValueError(f"Version {version} not found")

        self.metadata["active"][model_type] = {
            "name": model_name,
            "version": version,
            "path": matching[0]["path"]
        }

        self._save_metadata()
        logger.info(f"Set active {model_type} model to {model_name} {version}")

    def delete_version(self, model_type: str, model_name: str, version: str):
        """Delete a specific model version."""
        model_key = f"{model_type}/{model_name}"

        if model_key not in self.metadata["models"]:
            raise ValueError(f"Model not found: {model_key}")

        versions = self.metadata["models"][model_key]["versions"]
        matching = [v for v in versions if v["version"] == version]

        if not matching:
            raise ValueError(f"Version {version} not found")

        # Remove file
        model_path = Path(matching[0]["path"])
        if model_path.exists():
            model_path.unlink()

        # Update metadata
        self.metadata["models"][model_key]["versions"] = [
            v for v in versions if v["version"] != version
        ]

        # Clear active if this was active
        active = self.metadata["active"].get(model_type, {})
        if active.get("name") == model_name and active.get("version") == version:
            del self.metadata["active"][model_type]

        self._save_metadata()
        logger.info(f"Deleted {model_key} version {version}")


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry(base_path: str = "models/saved") -> ModelRegistry:
    """Get or create the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(base_path)
    return _registry


def load_model(
    model_type: str = "isolation_forest",
    model_name: Optional[str] = None,
    version: Optional[str] = None
) -> Union[IsolationForestDetector, LSTMAutoencoderDetector]:
    """Convenience function to load a model from the registry."""
    registry = get_registry()
    return registry.load(model_type, model_name, version)
