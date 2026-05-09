"""
Session Handler - Manages user session state for recommendations.

Tracks user sessions, maintains session context, and provides
session-based personalization signals.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import uuid
from shared.logger import get_logger

logger = get_logger(__name__)


class SessionData:
    """Represents a user session with its context."""
    
    def __init__(
        self, 
        session_id: str, 
        user_id: Optional[int] = None,
        ttl_minutes: int = 30
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.ttl_minutes = ttl_minutes
        self.interactions: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {
            'device_type': 'unknown',
            'referrer': None,
            'landing_page': None,
            'search_query': None,
            'filters_applied': [],
            'cart_items': [],
            'viewed_items': []
        }
        self.metadata: Dict[str, Any] = {}
    
    def is_active(self) -> bool:
        """Check if session is still active."""
        expiry = self.created_at + timedelta(minutes=self.ttl_minutes)
        return datetime.now() < expiry
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    def add_interaction(self, interaction: Dict[str, Any]) -> None:
        """Add an interaction to the session."""
        self.interactions.append({
            **interaction,
            'timestamp': datetime.now().isoformat()
        })
        self.update_activity()
        
        # Update context based on interaction
        interaction_type = interaction.get('type')
        item_id = interaction.get('item_id')
        
        if interaction_type == 'view' and item_id:
            if item_id not in self.context['viewed_items']:
                self.context['viewed_items'].append(item_id)
        elif interaction_type == 'add_to_cart' and item_id:
            if item_id not in self.context['cart_items']:
                self.context['cart_items'].append(item_id)
    
    def get_recent_interactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent interactions."""
        return self.interactions[-limit:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'is_active': self.is_active(),
            'interaction_count': len(self.interactions),
            'context': self.context,
            'metadata': self.metadata
        }


class SessionHandler:
    """
    Manages user sessions for recommendation personalization.
    
    Features:
    - Session creation and management
    - Session context tracking
    - Interaction history within session
    - Session-based recommendation signals
    - Automatic session expiration
    """
    
    DEFAULT_TTL_MINUTES = 30
    MAX_SESSIONS_PER_USER = 10
    
    def __init__(self, ttl_minutes: int = DEFAULT_TTL_MINUTES):
        """
        Initialize session handler.
        
        Args:
            ttl_minutes: Session time-to-live in minutes
        """
        self.ttl_minutes = ttl_minutes
        self._sessions: Dict[str, SessionData] = {}
        self._user_sessions: Dict[int, List[str]] = {}  # user_id -> session_ids
        
        logger.info(f"SessionHandler initialized with TTL={ttl_minutes} minutes")
    
    def create_session(
        self, 
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> SessionData:
        """
        Create a new session.
        
        Args:
            user_id: Optional user ID
            session_id: Optional custom session ID (generated if not provided)
            initial_context: Initial session context
            
        Returns:
            New SessionData object
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        session = SessionData(
            session_id=session_id,
            user_id=user_id,
            ttl_minutes=self.ttl_minutes
        )
        
        if initial_context:
            session.context.update(initial_context)
        
        self._sessions[session_id] = session
        
        # Track session by user
        if user_id is not None:
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = []
            self._user_sessions[user_id].append(session_id)
            
            # Limit sessions per user
            if len(self._user_sessions[user_id]) > self.MAX_SESSIONS_PER_USER:
                old_session_id = self._user_sessions[user_id].pop(0)
                if old_session_id in self._sessions:
                    del self._sessions[old_session_id]
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Get an existing session.
        
        Args:
            session_id: Session ID
            
        Returns:
            SessionData or None if not found/expired
        """
        if session_id not in self._sessions:
            logger.debug(f"Session {session_id} not found")
            return None
        
        session = self._sessions[session_id]
        if not session.is_active():
            logger.debug(f"Session {session_id} expired")
            self._remove_session(session_id)
            return None
        
        return session
    
    def update_session_context(
        self, 
        session_id: str, 
        context_updates: Dict[str, Any]
    ) -> bool:
        """
        Update session context.
        
        Args:
            session_id: Session ID
            context_updates: Dictionary of context updates
            
        Returns:
            True if successful
        """
        session = self.get_session(session_id)
        if session is None:
            return False
        
        session.context.update(context_updates)
        session.update_activity()
        logger.debug(f"Updated context for session {session_id}")
        return True
    
    def add_session_interaction(
        self,
        session_id: str,
        interaction_type: str,
        item_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add an interaction to a session.
        
        Args:
            session_id: Session ID
            interaction_type: Type of interaction (view, click, cart, etc.)
            item_id: Optional item ID
            metadata: Optional additional metadata
            
        Returns:
            True if successful
        """
        session = self.get_session(session_id)
        if session is None:
            return False
        
        interaction = {
            'type': interaction_type,
            'item_id': item_id,
            'metadata': metadata or {}
        }
        
        session.add_interaction(interaction)
        logger.debug(f"Added {interaction_type} interaction to session {session_id}")
        return True
    
    def get_user_active_sessions(self, user_id: int) -> List[SessionData]:
        """
        Get all active sessions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of active SessionData objects
        """
        if user_id not in self._user_sessions:
            return []
        
        active_sessions = []
        for session_id in self._user_sessions[user_id]:
            session = self.get_session(session_id)
            if session and session.is_active():
                active_sessions.append(session)
        
        return active_sessions
    
    def get_latest_session(self, user_id: int) -> Optional[SessionData]:
        """
        Get the most recent active session for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Most recent SessionData or None
        """
        sessions = self.get_user_active_sessions(user_id)
        if not sessions:
            return None
        
        return max(sessions, key=lambda s: s.last_activity)
    
    def _remove_session(self, session_id: str) -> None:
        """Remove a session."""
        if session_id not in self._sessions:
            return
        
        session = self._sessions[session_id]
        user_id = session.user_id
        
        del self._sessions[session_id]
        
        if user_id is not None and user_id in self._user_sessions:
            if session_id in self._user_sessions[user_id]:
                self._user_sessions[user_id].remove(session_id)
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove all expired sessions.
        
        Returns:
            Number of sessions removed
        """
        expired = [
            sid for sid, session in self._sessions.items()
            if not session.is_active()
        ]
        
        for session_id in expired:
            self._remove_session(session_id)
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        
        return len(expired)
    
    def get_session_signals(self, session_id: str) -> Dict[str, Any]:
        """
        Extract recommendation signals from a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary of recommendation signals
        """
        session = self.get_session(session_id)
        if session is None:
            return {}
        
        recent_interactions = session.get_recent_interactions(20)
        
        signals = {
            'viewed_items': session.context.get('viewed_items', [])[-10:],
            'cart_items': session.context.get('cart_items', []),
            'recent_item_ids': [
                i.get('item_id') for i in recent_interactions 
                if i.get('item_id')
            ][-10:],
            'interaction_types': [
                i.get('type') for i in recent_interactions
            ],
            'device_type': session.context.get('device_type', 'unknown'),
            'search_query': session.context.get('search_query'),
            'filters_applied': session.context.get('filters_applied', [])
        }
        
        return signals
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session handler statistics."""
        active_count = sum(1 for s in self._sessions.values() if s.is_active())
        
        return {
            'total_sessions': len(self._sessions),
            'active_sessions': active_count,
            'unique_users': len(self._user_sessions),
            'avg_interactions_per_session': (
                sum(len(s.interactions) for s in self._sessions.values()) / 
                len(self._sessions) if self._sessions else 0
            )
        }
