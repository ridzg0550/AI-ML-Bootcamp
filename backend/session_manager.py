"""
Session manager for DungeonBrain++ instances.
Handles lifecycle of game sessions with auto-cleanup on inactivity.
"""

import os
import sys
import time
import uuid
from typing import Dict, Optional

# Add parent directory so we can import main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import DungeonBrainPlus

PERSIST_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dungeonbrain_sessions")
MAX_SESSIONS = 5
SESSION_TIMEOUT = 1800  # 30 minutes


class SessionManager:
    """Manages multiple DungeonBrain++ game sessions."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.sessions: Dict[str, DungeonBrainPlus] = {}
        self.last_activity: Dict[str, float] = {}
        self.opening_messages: Dict[str, str] = {}
        os.makedirs(PERSIST_BASE, exist_ok=True)

    def create_session(self) -> tuple[str, str]:
        """Create a new game session. Returns (session_id, opening_message)."""
        self._cleanup_stale()

        if len(self.sessions) >= MAX_SESSIONS:
            # Evict oldest inactive session
            oldest_id = min(self.last_activity, key=self.last_activity.get)
            self.end_session(oldest_id)

        session_id = uuid.uuid4().hex[:12]
        persist_dir = os.path.join(PERSIST_BASE, session_id)
        os.makedirs(persist_dir, exist_ok=True)

        brain = DungeonBrainPlus(api_key=self.api_key, persist_dir=persist_dir)

        opening = (
            "You stand at the entrance of a misty forest. A worn path leads deeper "
            "into the trees, and you hear the distant sound of running water. "
            "Ancient runes are carved into a moss-covered stone beside the trail."
        )
        brain.memory.add(f"[Turn 0] DM: {opening}", is_critical=True)

        self.sessions[session_id] = brain
        self.last_activity[session_id] = time.time()
        self.opening_messages[session_id] = opening

        return session_id, opening

    def get_session(self, session_id: str) -> Optional[DungeonBrainPlus]:
        """Get an active session by ID."""
        if session_id in self.sessions:
            self.last_activity[session_id] = time.time()
            return self.sessions[session_id]
        return None

    def end_session(self, session_id: str) -> bool:
        """End a session, save state, and free resources."""
        if session_id not in self.sessions:
            return False
        try:
            self.sessions[session_id].save_state()
        except Exception:
            pass
        del self.sessions[session_id]
        del self.last_activity[session_id]
        self.opening_messages.pop(session_id, None)
        return True

    def save_session(self, session_id: str) -> bool:
        """Persist session state to disk."""
        brain = self.get_session(session_id)
        if brain is None:
            return False
        brain.save_state()
        return True

    def _cleanup_stale(self):
        """Remove sessions that have been inactive for too long."""
        now = time.time()
        stale = [sid for sid, t in self.last_activity.items() if now - t > SESSION_TIMEOUT]
        for sid in stale:
            self.end_session(sid)

    @property
    def active_count(self) -> int:
        return len(self.sessions)
