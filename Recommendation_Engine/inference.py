"""
Recommendation Inference Engine

Inference pipeline for:
- User embedding generation
- Candidate retrieval
- DLRM/XGBoost/LightGBM ranking
- Re-ranking optimization
- Final recommendation generation
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from shared.logger import get_logger
from .model_registry import ModelRegistry

logger = get_logger(__name__)


class RecommendationInference:
    """
    Inference engine for generating recommendations.
    
    Pipeline:
    1. Load trained models
    2. Generate user embeddings
    3. Retrieve candidates
    4. Rank candidates
    5. Apply re-ranking
    6. Return final recommendations
    """
    
    def __init__(
        self,
        models_dir: str = "output/models",
        device: str = "cpu"
    ):
        self.models_dir = Path(models_dir)
        self.device = device
        
        self.registry = ModelRegistry(models_dir)
        
        # Loaded models
        self.three_tower_model = None
        self.dlrm_model = None
        self.xgboost_model = None
        self.lightgbm_model = None
        
        # Embeddings
        self.user_embeddings = None
        self.item_embeddings = None
        self.user_ids = None
        self.item_ids = None
        
        # Item metadata for explanations
        self.item_metadata = None
    
    def load_models(
        self,
        three_tower_id: Optional[str] = None,
        dlrm_id: Optional[str] = None,
        xgboost_id: Optional[str] = None,
        lightgbm_id: Optional[str] = None
    ):
        """
        Load models from registry.
        
        Args:
            three_tower_id: Three-Tower model ID (uses active if not provided)
            dlrm_id: DLRM model ID (uses active if not provided)
            xgboost_id: XGBoost model ID (uses active if not provided)
            lightgbm_id: LightGBM model ID (uses active if not provided)
        """
        logger.info("Loading models for inference...")
        
        # Load Three-Tower model
        if three_tower_id is None:
            three_tower_id = self.registry.get_active_model('three_tower')
        
        if three_tower_id:
            logger.info(f"Loading Three-Tower model: {three_tower_id}")
            # Note: In production, you'd need to reconstruct the architecture
            # This is a simplified version
            pass
        
        # Load DLRM model
        if dlrm_id is None:
            dlrm_id = self.registry.get_active_model('dlrm')
        
        if dlrm_id:
            try:
                self.dlrm_model = self.registry.load_model(dlrm_id)
                logger.info(f"Loaded DLRM model: {dlrm_id}")
            except Exception as e:
                logger.warning(f"Could not load DLRM model: {e}")
        
        # Load XGBoost model
        if xgboost_id is None:
            xgboost_id = self.registry.get_active_model('xgboost')
        
        if xgboost_id:
            try:
                self.xgboost_model = self.registry.load_model(xgboost_id)
                logger.info(f"Loaded XGBoost model: {xgboost_id}")
            except Exception as e:
                logger.warning(f"Could not load XGBoost model: {e}")
        
        # Load LightGBM model
        if lightgbm_id is None:
            lightgbm_id = self.registry.get_active_model('lightgbm')
        
        if lightgbm_id:
            try:
                self.lightgbm_model = self.registry.load_model(lightgbm_id)
                logger.info(f"Loaded LightGBM model: {lightgbm_id}")
            except Exception as e:
                logger.warning(f"Could not load LightGBM model: {e}")
        
        # Load embeddings
        try:
            self.user_embeddings, self.user_ids = self.registry.load_embeddings('user')
            logger.info(f"Loaded user embeddings: {self.user_embeddings.shape}")
        except Exception as e:
            logger.warning(f"Could not load user embeddings: {e}")
        
        try:
            self.item_embeddings, self.item_ids = self.registry.load_embeddings('item')
            logger.info(f"Loaded item embeddings: {self.item_embeddings.shape}")
        except Exception as e:
            logger.warning(f"Could not load item embeddings: {e}")
    
    def load_item_metadata(self, items_df: pd.DataFrame):
        """Load item metadata for explanations."""
        self.item_metadata = items_df.set_index('item_id')
        logger.info(f"Loaded item metadata for {len(items_df)} items")
    
    def generate_user_embedding(
        self,
        user_features: np.ndarray,
        user_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embedding for a user.
        
        Args:
            user_features: User feature vector
            user_id: Optional user ID
            
        Returns:
            User embedding vector
        """
        if not TORCH_AVAILABLE or self.three_tower_model is None:
            # Fallback: use average of interacted item embeddings
            if self.item_embeddings is not None:
                return self.item_embeddings.mean(axis=0)
            return np.zeros(64)  # Default embedding dim
        
        with torch.no_grad():
            user_tensor = torch.FloatTensor(user_features).unsqueeze(0).to(self.device)
            # Simplified - would need full model forward pass in production
            pass
        
        return np.zeros(64)
    
    def retrieve_candidates(
        self,
        user_embedding: np.ndarray,
        top_k: int = 100,
        exclude_items: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieve candidate items using embedding similarity.
        
        Args:
            user_embedding: User embedding vector
            top_k: Number of candidates to retrieve
            exclude_items: Items to exclude (already purchased, etc.)
            
        Returns:
            List of (item_id, score) tuples
        """
        if self.item_embeddings is None:
            logger.warning("No item embeddings available")
            return []
        
        # Compute cosine similarity
        user_emb_norm = user_embedding / (np.linalg.norm(user_embedding) + 1e-8)
        item_embs_norm = self.item_embeddings / (np.linalg.norm(self.item_embeddings, axis=1, keepdims=True) + 1e-8)
        
        scores = np.dot(item_embs_norm, user_emb_norm)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            item_id = self.item_ids[idx] if self.item_ids else f"item_{idx}"
            
            # Skip excluded items
            if exclude_items and item_id in exclude_items:
                continue
            
            results.append((item_id, float(scores[idx])))
        
        return results
    
    def rank_candidates(
        self,
        user_features: np.ndarray,
        candidates: List[Tuple[str, float]],
        item_features_dict: Dict[str, np.ndarray]
    ) -> List[Tuple[str, float]]:
        """
        Rank candidates using DLRM or fallback models.
        
        Args:
            user_features: User feature vector
            candidates: List of (item_id, retrieval_score) tuples
            item_features_dict: Dict mapping item_id to item features
            
        Returns:
            List of (item_id, rank_score) tuples
        """
        ranked = []
        
        for item_id, retrieval_score in candidates:
            if item_id not in item_features_dict:
                continue
            
            item_features = item_features_dict[item_id]
            
            # Try DLRM first
            if self.dlrm_model is not None and TORCH_AVAILABLE:
                try:
                    combined_features = np.hstack([user_features, item_features])
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(combined_features).unsqueeze(0)
                        score = self.dlrm_model(features_tensor).sigmoid().item()
                    ranked.append((item_id, score))
                    continue
                except Exception as e:
                    logger.warning(f"DLRM scoring failed: {e}")
            
            # Try XGBoost
            if self.xgboost_model is not None:
                try:
                    import xgboost as xgb
                    combined_features = np.hstack([user_features, item_features])
                    dmatrix = xgb.DMatrix(combined_features.reshape(1, -1))
                    score = self.xgboost_model.predict(dmatrix)[0]
                    ranked.append((item_id, float(score)))
                    continue
                except Exception as e:
                    logger.warning(f"XGBoost scoring failed: {e}")
            
            # Try LightGBM
            if self.lightgbm_model is not None:
                try:
                    combined_features = np.hstack([user_features, item_features])
                    score = self.lightgbm_model.predict(combined_features.reshape(1, -1))[0]
                    ranked.append((item_id, float(score)))
                    continue
                except Exception as e:
                    logger.warning(f"LightGBM scoring failed: {e}")
            
            # Fallback to retrieval score
            ranked.append((item_id, retrieval_score))
        
        # Sort by score descending
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def apply_reranking(
        self,
        ranked_candidates: List[Tuple[str, float]],
        user_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply re-ranking for diversity, freshness, business rules.
        
        Args:
            ranked_candidates: List of (item_id, score) tuples
            user_id: User ID for personalization
            config: Re-ranking configuration
            
        Returns:
            List of recommendation dictionaries
        """
        config = config or {}
        
        # Import reranking components
        from .ReRanking.reranker import ReRanker
        
        reranker = ReRanker(config=config)
        
        # Extract item IDs and scores
        item_ids = [c[0] for c in ranked_candidates]
        scores = np.array([c[1] for c in ranked_candidates])
        
        # Apply reranking
        reranked_indices = reranker.rerank(
            item_ids=item_ids,
            scores=scores,
            user_id=user_id
        )
        
        # Build final recommendations
        recommendations = []
        for rank, idx in enumerate(reranked_indices[:10]):  # Top 10
            item_id = item_ids[idx]
            
            rec = {
                'item_id': item_id,
                'ranking_score': float(scores[idx]),
                'rerank_score': float(scores[idx] * (1.0 - rank * 0.05)),  # Simple decay
                'rank': rank + 1,
                'recommendation_reason': self._generate_reason(item_id, rank),
                'applied_business_rules': []
            }
            
            # Add business rule explanations
            if config.get('diversity_boost', False):
                rec['applied_business_rules'].append('diversity_optimization')
            if config.get('freshness_boost', False):
                rec['applied_business_rules'].append('freshness_boost')
            if config.get('margin_boost', False):
                rec['applied_business_rules'].append('margin_boost')
            
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_reason(self, item_id: str, rank: int) -> str:
        """Generate explanation for recommendation."""
        reasons = [
            "Based on your purchase history",
            "Popular in your category",
            "Trending now",
            "Highly rated by similar users",
            "New arrival matching your preferences"
        ]
        
        if self.item_metadata is not None and item_id in self.item_metadata.index:
            item_info = self.item_metadata.loc[item_id]
            if 'category' in item_info:
                return f"Recommended from {item_info['category']} category"
        
        return reasons[min(rank, len(reasons) - 1)]
    
    def recommend(
        self,
        user_id: str,
        user_features: np.ndarray,
        item_features_dict: Dict[str, np.ndarray],
        exclude_items: Optional[List[str]] = None,
        top_k: int = 10,
        config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Full recommendation pipeline for a user.
        
        Args:
            user_id: User ID
            user_features: User feature vector
            item_features_dict: Dict mapping item_id to features
            exclude_items: Items to exclude
            top_k: Number of recommendations
            config: Configuration dict
            
        Returns:
            List of recommendation dictionaries
        """
        logger.info(f"Generating recommendations for user {user_id}")
        
        # Step 1: Generate user embedding
        user_embedding = self.generate_user_embedding(user_features)
        
        # Step 2: Retrieve candidates
        candidates = self.retrieve_candidates(
            user_embedding,
            top_k=50,  # Retrieve more than needed
            exclude_items=exclude_items
        )
        
        if not candidates:
            logger.warning("No candidates retrieved")
            return []
        
        # Step 3: Rank candidates
        ranked = self.rank_candidates(
            user_features,
            candidates,
            item_features_dict
        )
        
        # Step 4: Apply re-ranking
        recommendations = self.apply_reranking(
            ranked,
            user_id,
            config
        )
        
        # Return top-k
        return recommendations[:top_k]
    
    def batch_recommend(
        self,
        users_df: pd.DataFrame,
        item_features_dict: Dict[str, np.ndarray],
        top_k: int = 10,
        config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Generate recommendations for multiple users.
        
        Args:
            users_df: DataFrame with user features
            item_features_dict: Dict mapping item_id to features
            top_k: Number of recommendations per user
            config: Configuration dict
            
        Returns:
            DataFrame with all recommendations
        """
        all_recommendations = []
        
        for _, row in users_df.iterrows():
            user_id = row['user_id']
            user_features = row.drop('user_id').values
            
            recs = self.recommend(
                user_id=user_id,
                user_features=user_features,
                item_features_dict=item_features_dict,
                top_k=top_k,
                config=config
            )
            
            for rec in recs:
                rec['user_id'] = user_id
                all_recommendations.append(rec)
        
        return pd.DataFrame(all_recommendations)
    
    def export_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        output_path: str,
        format: str = "json"
    ):
        """Export recommendations to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_file, 'w') as f:
                json.dump(recommendations, f, indent=2)
        elif format == "parquet":
            df = pd.DataFrame(recommendations)
            df.to_parquet(output_file, index=False)
        elif format == "csv":
            df = pd.DataFrame(recommendations)
            df.to_csv(output_file, index=False)
        
        logger.info(f"Exported {len(recommendations)} recommendations to {output_path}")
