"""
HTTP chat session manager.

Manages conversation state for /chat endpoint.
Each session tracks message history, handles memory context injection,
and manages turn persistence/summarization.
"""

import logging
import os
import threading
import time
import uuid
from typing import Optional

from .config import get_config

logger = logging.getLogger(__name__)

# In-memory session store
_sessions = {}
_sessions_lock = threading.Lock()


class ChatSession:
    """A single chat session with conversation history."""

    def __init__(self, session_id: str, username: str):
        self.session_id = session_id
        self.username = username
        self.messages = []  # [{"role": "system"|"user"|"assistant", "content": "..."}]
        self.unsummarized_turns = []  # [(user_text, assistant_text)]
        self.last_activity = time.time()
        self.created_at = time.time()

    def add_user_message(self, text: str):
        self.messages.append({"role": "user", "content": text})
        self.last_activity = time.time()

    def add_assistant_message(self, text: str):
        self.messages.append({"role": "assistant", "content": text})
        self.unsummarized_turns.append((
            self.messages[-2]["content"],  # user
            text  # assistant
        ))
        self.last_activity = time.time()

    def idle_seconds(self) -> float:
        return time.time() - self.last_activity

    def build_messages(self, system_prompt: str, memory_context: str) -> list:
        """Build full message array for inference request."""
        result = []

        # System prompt with memory context
        system_content = system_prompt
        if memory_context:
            system_content += "\n\n## Relevant memories:\n\n" + memory_context

        if system_content:
            result.append({"role": "system", "content": system_content})

        # Conversation history (bounded)
        cfg = get_config()
        max_turns = cfg.get("chat_max_turns_stored", 200)
        # Keep last N messages (each turn = 2 messages)
        history = self.messages[-(max_turns * 2):]
        result.extend(history)

        return result


def get_or_create_session(session_id: Optional[str], username: str) -> ChatSession:
    """Get existing session or create a new one."""
    with _sessions_lock:
        if session_id and session_id in _sessions:
            session = _sessions[session_id]
            session.last_activity = time.time()
            return session

        new_id = session_id or str(uuid.uuid4())
        session = ChatSession(new_id, username)
        _sessions[new_id] = session
        return session


def load_workspace_files() -> str:
    """Load persona/workspace files for system prompt.
    
    SOUL.md priority:
    - Single-user mode: ~/.ragger/SOUL.md first, fall back to /var/ragger/SOUL.md
    - Multi-user mode: /var/ragger/SOUL.md only (no user fallback)
    
    Other files (USER.md, MEMORY.md): user dir only
    Shared files (AGENTS.md, TOOLS.md): follow same priority as SOUL.md
    """
    from .config import get_config
    cfg = get_config()
    
    user_dir = os.path.expanduser("~/.ragger")
    common_dir = "/var/ragger"
    
    # Multi-user mode requires SOUL.md in common dir
    if not cfg["single_user"]:
        soul_path = os.path.join(common_dir, "SOUL.md")
        if not os.path.exists(soul_path):
            raise RuntimeError(
                f"Multi-user mode requires SOUL.md in {common_dir}\n"
                "SOUL.md defines the assistant's personality and must be present for consistent behavior."
            )
    
    # (filename, allow_common)
    file_specs = [
        ("SOUL.md", True),    # Allow common dir
        ("USER.md", False),   # User only
        ("AGENTS.md", True),  # Allow common dir
        ("TOOLS.md", True),   # Allow common dir
    ]
    
    parts = []
    
    for fname, allow_common in file_specs:
        fpath = None
        
        if cfg["single_user"]:
            # Single-user mode: user dir first, fall back to common
            user_path = os.path.join(user_dir, fname)
            if os.path.exists(user_path):
                fpath = user_path
            elif allow_common:
                common_path = os.path.join(common_dir, fname)
                if os.path.exists(common_path):
                    fpath = common_path
        else:
            # Multi-user mode: common dir first, NO fallback for SOUL.md
            if allow_common:
                common_path = os.path.join(common_dir, fname)
                if os.path.exists(common_path):
                    fpath = common_path
            
            # For non-SOUL files with allow_common, fall back to user
            # For SOUL.md in multi-user mode, stop here (no fallback)
            if not fpath and (fname != "SOUL.md" or not allow_common):
                user_path = os.path.join(user_dir, fname)
                if os.path.exists(user_path):
                    fpath = user_path
        
        if fpath:
            try:
                with open(fpath) as f:
                    content = f.read().strip()
                if content:
                    parts.append(f"## {fname}\n\n{content}")
            except Exception:
                pass

    return "\n\n---\n\n".join(parts) if parts else ""


def cleanup_expired_sessions(memory=None, inference_client=None):
    """Remove expired sessions, optionally summarizing their turns."""
    cfg = get_config()
    pause_minutes = cfg.get("chat_pause_minutes", 10)

    with _sessions_lock:
        expired = []
        for sid, session in _sessions.items():
            if session.idle_seconds() > pause_minutes * 60:
                expired.append(sid)

        for sid in expired:
            session = _sessions.pop(sid)
            if session.unsummarized_turns and memory and inference_client:
                # Background summarize
                _bg_summarize(session, memory, inference_client)


def _bg_summarize(session, memory, inference_client):
    """Summarize session turns in background thread."""
    turns = list(session.unsummarized_turns)
    if not turns:
        return

    def _do_summarize():
        try:
            # Build conversation text
            conv_text = ""
            for user_text, assistant_text in turns:
                conv_text += f"**User:** {user_text}\n\n"
                conv_text += f"**Assistant:** {assistant_text}\n\n"

            messages = [
                {"role": "system", "content": "Summarize this conversation into a concise memory entry. "
                 "Extract key facts, decisions, and action items. Be brief but complete."},
                {"role": "user", "content": conv_text}
            ]

            response = inference_client.chat(messages=messages, stream=False)
            summary = inference_client.extract_content(response)

            if summary:
                memory.store(summary, {
                    "collection": "memory",
                    "category": "session-summary",
                    "source": f"chat-session-{session.session_id[:8]}"
                })
                logger.info(f"Summarized session {session.session_id[:8]} ({len(turns)} turns)")
        except Exception as e:
            logger.error(f"Session summarization failed: {e}")

    thread = threading.Thread(target=_do_summarize, daemon=True)
    thread.start()
