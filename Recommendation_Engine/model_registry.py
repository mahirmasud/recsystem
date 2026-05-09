"""
Model Registry - Centralized model storage and versioning system.

Manages:
- Model artifacts (PyTorch, XGBoost, LightGBM, sklearn)
- Embedding matrices
- Model metadata and versioning
- Model loading/saving
- Experiment tracking
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from shared.logger import get_logger
from shared.config import Config

logger = get_logger(__name__)


class ModelMetadata:
    """Metadata for a registered model."""
    
    def __init__(
        self,
        model_id: str,
        model_type: str,
        version: str,
        created_at: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        training_data_hash: str,
        description: str = ""
    ):
        self.model_id = model_id
        self.model_type = model_type
        self.version = version
        self.created_at = created_at
        self.metrics = metrics
        self.hyperparameters = hyperparameters
        self.training_data_hash = training_data_hash
        self.description = description
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "version": self.version,
            "created_at": self.created_at,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "training_data_hash": self.training_data_hash,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        return cls(
            model_id=data["model_id"],
            model_type=data["model_type"],
            version=data["version"],
            created_at=data["created_at"],
            metrics=data.get("metrics", {}),
            hyperparameters=data.get("hyperparameters", {}),
            training_data_hash=data.get("training_data_hash", ""),
            description=data.get("description", "")
        )


class ModelRegistry:
    """
    Centralized registry for managing recommendation models.
    
    Features:
    - Version control for models
    - Metadata tracking
    - Multiple model type support (PyTorch, XGBoost, LightGBM, sklearn)
    - Embedding storage
    - Experiment tracking
    """
    
    def __init__(self, models_dir: str = "output/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for different model types
        self.pytorch_dir = self.models_dir / "pytorch"
        self.xgboost_dir = self.models_dir / "xgboost"
        self.lightgbm_dir = self.models_dir / "lightgbm"
        self.sklearn_dir = self.models_dir / "sklearn"
        self.embeddings_dir = self.models_dir / "embeddings"
        self.metadata_dir = self.models_dir / "metadata"
        
        for dir_path in [self.pytorch_dir, self.xgboost_dir, 
                         self.lightgbm_dir, self.sklearn_dir,
                         self.embeddings_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory registry
        self.registered_models: Dict[str, ModelMetadata] = {}
        self.active_models: Dict[str, str] = {}  # model_type -> model_id
        
        # Load existing registry
        self._load_registry()
    
    def _load_registry(self):
        """Load existing model registry from disk."""
        registry_file = self.metadata_dir / "registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                data = json.load(f)
                self.registered_models = {
                    k: ModelMetadata.from_dict(v) 
                    for k, v in data.get("models", {}).items()
                }
                self.active_models = data.get("active_models", {})
            logger.info(f"Loaded registry with {len(self.registered_models)} models")
    
    def _save_registry(self):
        """Save model registry to disk."""
        registry_file = self.metadata_dir / "registry.json"
        data = {
            "models": {k: v.to_dict() for k, v in self.registered_models.items()},
            "active_models": self.active_models
        }
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of training data for versioning."""
        # Use first 1000 rows for speed
        sample = df.head(1000)
        data_str = sample.to_csv(index=False).encode('utf-8')
        return hashlib.md5(data_str).hexdigest()
    
    def register_model(
        self,
        model: Any,
        model_type: str,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_data: Optional[pd.DataFrame] = None,
        description: str = ""
    ) -> str:
        """
        Register a trained model in the registry.
        
        Args:
            model: Trained model object
            model_type: Type of model ('three_tower', 'dlrm', 'xgboost', 'lightgbm', etc.)
            model_id: Unique model ID (auto-generated if not provided)
            version: Version string (auto-generated if not provided)
            metrics: Evaluation metrics
            hyperparameters: Model hyperparameters
            training_data: Training DataFrame for hashing
            description: Model description
            
        Returns:
            model_id: The registered model ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if model_id is None:
            model_id = f"{model_type}_{timestamp}"
        
        if version is None:
            version = "1.0.0"
        
        # Compute data hash if training data provided
        training_data_hash = ""
        if training_data is not None:
            training_data_hash = self._compute_data_hash(training_data)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=model_type,
            version=version,
            created_at=timestamp,
            metrics=metrics or {},
            hyperparameters=hyperparameters or {},
            training_data_hash=training_data_hash,
            description=description
        )
        
        # Save model based on type
        model_path = self._save_model(model, model_type, model_id)
        metadata.hyperparameters["model_path"] = str(model_path)
        
        # Register
        self.registered_models[model_id] = metadata
        self._save_registry()
        
        logger.info(f"Registered model {model_id} at {model_path}")
        return model_id
    
    def _save_model(self, model: Any, model_type: str, model_id: str) -> Path:
        """Save model to appropriate directory based on type."""
        
        if model_type in ['three_tower', 'dlrm', 'user_tower', 'item_tower', 'context_tower']:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required for deep learning models")
            model_path = self.pytorch_dir / f"{model_id}.pt"
            torch.save(model.state_dict(), model_path)
            
        elif model_type == 'xgboost':
            model_path = self.xgboost_dir / f"{model_id}.json"
            model.save_model(str(model_path))
            
        elif model_type == 'lightgbm':
            model_path = self.lightgbm_dir / f"{model_id}.pkl"
            joblib.dump(model, model_path)
            
        else:  # sklearn or other
            model_path = self.sklearn_dir / f"{model_id}.pkl"
            joblib.dump(model, model_path)
        
        return model_path
    
    def load_model(self, model_id: str) -> Any:
        """
        Load a model from the registry.
        
        Args:
            model_id: ID of model to load
            
        Returns:
            Loaded model object
        """
        if model_id not in self.registered_models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self.registered_models[model_id]
        model_type = metadata.model_type
        model_path = Path(metadata.hyperparameters.get("model_path", ""))
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model {model_id} from {model_path}")
        
        if model_type in ['three_tower', 'dlrm', 'user_tower', 'item_tower', 'context_tower']:
            # Need to reconstruct model architecture first
            # This is a simplified load - in production, you'd need model config
            raise NotImplementedError(
                "PyTorch models require architecture reconstruction. "
                "Use load_model_with_architecture() instead."
            )
            
        elif model_type == 'xgboost':
            try:
                import xgboost as xgb
                model = xgb.XGBRanker()
                model.load_model(str(model_path))
                return model
            except ImportError:
                raise ImportError("XGBoost required for xgboost models")
                
        elif model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                return joblib.load(model_path)
            except ImportError:
                raise ImportError("LightGBM required for lightgbm models")
                
        else:
            return joblib.load(model_path)
    
    def set_active_model(self, model_type: str, model_id: str):
        """Set the active model for a given model type."""
        if model_id not in self.registered_models:
            raise ValueError(f"Model {model_id} not found")
        
        self.active_models[model_type] = model_id
        self._save_registry()
        logger.info(f"Set active {model_type} model to {model_id}")
    
    def get_active_model(self, model_type: str) -> Optional[str]:
        """Get the active model ID for a given model type."""
        return self.active_models.get(model_type)
    
    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model."""
        return self.registered_models.get(model_id)
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered models, optionally filtered by type."""
        models = list(self.registered_models.values())
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        return [m.to_dict() for m in models]
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        embedding_type: str,
        ids: Optional[List[str]] = None,
        model_id: Optional[str] = None
    ) -> Path:
        """
        Save embedding matrix to disk.
        
        Args:
            embeddings: Numpy array of embeddings (n_items x embedding_dim)
            embedding_type: Type of embeddings ('user', 'item', 'context')
            ids: Optional list of IDs corresponding to embeddings
            model_id: Optional model ID that generated these embeddings
            
        Returns:
            Path to saved embeddings
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{embedding_type}_embeddings_{timestamp}"
        if model_id:
            filename = f"{embedding_type}_embeddings_{model_id}"
        
        # Save embeddings as parquet for efficient loading
        if ids is not None:
            df = pd.DataFrame(embeddings)
            df.insert(0, 'id', ids)
        else:
            df = pd.DataFrame(embeddings)
            df.insert(0, 'id', range(len(embeddings)))
        
        path = self.embeddings_dir / f"{filename}.parquet"
        df.to_parquet(path, index=False)
        
        logger.info(f"Saved {embedding_type} embeddings ({embeddings.shape}) to {path}")
        return path
    
    def load_embeddings(
        self,
        embedding_type: str,
        model_id: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        """
        Load embeddings from disk.
        
        Args:
            embedding_type: Type of embeddings to load
            model_id: Optional model ID filter
            
        Returns:
            Tuple of (embeddings array, optional IDs list)
        """
        # Find most recent embeddings of this type
        pattern = f"{embedding_type}_embeddings_*.parquet"
        files = list(self.embeddings_dir.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No {embedding_type} embeddings found")
        
        # Sort by modification time, get latest
        latest_file = max(files, key=lambda p: p.stat().st_mtime)
        
        df = pd.read_parquet(latest_file)
        ids = df['id'].tolist()
        embeddings = df.drop(columns=['id']).values
        
        logger.info(f"Loaded {embedding_type} embeddings ({embeddings.shape}) from {latest_file}")
        return embeddings, ids
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model from the registry."""
        if model_id not in self.registered_models:
            return False
        
        metadata = self.registered_models[model_id]
        model_path = Path(metadata.hyperparameters.get("model_path", ""))
        
        if model_path.exists():
            model_path.unlink()
        
        del self.registered_models[model_id]
        
        # Remove from active models if present
        for model_type, active_id in list(self.active_models.items()):
            if active_id == model_id:
                del self.active_models[model_type]
        
        self._save_registry()
        logger.info(f"Deleted model {model_id}")
        return True
    
    def export_registry(self, output_path: str):
        """Export full registry to JSON file."""
        data = {
            "exported_at": datetime.now().isoformat(),
            "total_models": len(self.registered_models),
            "active_models": self.active_models,
            "models": {k: v.to_dict() for k, v in self.registered_models.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported registry to {output_path}")
