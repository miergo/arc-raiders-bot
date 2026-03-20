"""In-memory LRU session store for multi-turn conversation histories."""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict

from arc_raider_bot.config import MAX_SESSIONS

_sessions: OrderedDict[str, list] = OrderedDict()
_lock = threading.Lock()


def get_history(session_id: str | None) -> tuple[str, list]:
    """Return (session_id, message_history). Creates a new session if needed."""
    with _lock:
        if session_id and session_id in _sessions:
            _sessions.move_to_end(session_id)
            return session_id, _sessions[session_id]

        sid = session_id or uuid.uuid4().hex
        _sessions[sid] = []
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
        return sid, _sessions[sid]


def save_history(session_id: str, messages: list) -> None:
    """Persist updated message list back into the session store."""
    with _lock:
        _sessions[session_id] = messages
        _sessions.move_to_end(session_id)
