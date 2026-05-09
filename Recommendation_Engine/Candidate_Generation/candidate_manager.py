"""Candidate Manager - Orchestrates multiple candidate generation strategies."""
import numpy as np
from typing import Dict, Any, List, Optional
import logging
logger = logging.getLogger(__name__)

class CandidateManager:
    """Manages multiple candidate sources and merges results."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.candidate_sources = {}
        self.merge_weights = config.get('merge_weights', {
            'collaborative': 0.3,
            'embedding': 0.4,
            'popularity': 0.3
        })
        
    def register_source(self, name: str, source):
        """Register a candidate generation source."""
        self.candidate_sources[name] = source
        logger.info(f"Registered candidate source: {name}")
        
    def generate_candidates(self, user_id: int, user_embedding: Optional[np.ndarray] = None,
                           n_candidates: int = 500) -> List[Dict[str, Any]]:
        """Generate candidates from all sources and merge."""
        all_candidates = {}
        
        # Get candidates from each source
        if 'collaborative' in self.candidate_sources:
            cf_cands = self.candidate_sources['collaborative'].recommend(user_id, n=200)
            for item_id in cf_cands:
                if item_id not in all_candidates:
                    all_candidates[item_id] = {'score': 0.0, 'sources': []}
                all_candidates[item_id]['score'] += self.merge_weights.get('collaborative', 0.3)
                all_candidates[item_id]['sources'].append('collaborative')
        
        if 'embedding' in self.candidate_sources and user_embedding is not None:
            emb_cands = self.candidate_sources['embedding'].retrieve_by_similarity(user_embedding, k=200)
            for item_id in emb_cands:
                if item_id not in all_candidates:
                    all_candidates[item_id] = {'score': 0.0, 'sources': []}
                all_candidates[item_id]['score'] += self.merge_weights.get('embedding', 0.4)
                all_candidates[item_id]['sources'].append('embedding')
        
        if 'popularity' in self.candidate_sources:
            pop_cands = self.candidate_sources['popularity'].get_candidates(n=200)
            for item_id in pop_cands:
                if item_id not in all_candidates:
                    all_candidates[item_id] = {'score': 0.0, 'sources': []}
                all_candidates[item_id]['score'] += self.merge_weights.get('popularity', 0.3) * self.candidate_sources['popularity'].get_score(item_id)
                all_candidates[item_id]['sources'].append('popularity')
        
        # Sort by combined score
        sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1]['score'], reverse=True)
        
        return [{'item_id': item_id, **info} for item_id, info in sorted_candidates[:n_candidates]]
