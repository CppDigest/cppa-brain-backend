"""
Chat History Manager for RAG System
Handles per-client chat history with context management and conversation flow.
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from loguru import logger
import uuid


@dataclass
class ChatMessage:
    """Individual chat message."""
    message_id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ChatSession:
    """Chat session for a client."""
    session_id: str
    client_id: str
    created_at: str
    last_activity: str
    messages: List[ChatMessage]
    context: Dict[str, Any]
    max_history: int = 50
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


class ChatHistoryManager:
    """Manages chat history for multiple clients."""
    
    def __init__(self, max_sessions: int = 1000, session_timeout_hours: int = 24):
        """Initialize chat history manager."""
        self.logger = logger.bind(name="ChatHistoryManager")
        self.max_sessions = max_sessions
        self.session_timeout_hours = session_timeout_hours
        
        # Storage: client_id -> ChatSession
        self.sessions: Dict[str, ChatSession] = {}
        
        # Statistics
        self.total_messages = 0
        self.active_sessions = 0
        
        self.logger.info("ChatHistoryManager initialized")
    
    def create_session(self, client_id: str, max_history: int = 50) -> str:
        """Create a new chat session for a client."""
        try:
            # Check if session already exists
            if client_id in self.sessions:
                self.logger.info(f"Session already exists for client: {client_id}")
                return self.sessions[client_id].session_id
            
            # Create new session
            session_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            session = ChatSession(
                session_id=session_id,
                client_id=client_id,
                created_at=now,
                last_activity=now,
                messages=[],
                context={},
                max_history=max_history
            )
            
            self.sessions[client_id] = session
            self.active_sessions += 1
            
            self.logger.info(f"Created new session for client: {client_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Error creating session for client {client_id}: {e}")
            return None
    
    def add_message(self, client_id: str, role: str, content: str, 
                   metadata: Dict[str, Any] = None) -> str:
        """Add a message to a client's chat history."""
        try:
            if client_id not in self.sessions:
                # Auto-create session if it doesn't exist
                self.create_session(client_id)
            
            session = self.sessions[client_id]
            
            # Create message
            message_id = str(uuid.uuid4())
            message = ChatMessage(
                message_id=message_id,
                role=role,
                content=content,
                timestamp=datetime.now().isoformat(),
                metadata=metadata or {}
            )
            
            # Add to session
            session.messages.append(message)
            session.last_activity = datetime.now().isoformat()
            
            # Maintain max history limit
            if len(session.messages) > session.max_history:
                session.messages = session.messages[-session.max_history:]
            
            self.total_messages += 1
            self.logger.debug(f"Added {role} message to session {client_id}")
            
            return message_id
            
        except Exception as e:
            self.logger.error(f"Error adding message to session {client_id}: {e}")
            return None
    
    def get_chat_history(self, client_id: str, max_messages: int = None, 
                        include_metadata: bool = False) -> List[Dict[str, Any]]:
        """Get chat history for a client."""
        try:
            if client_id not in self.sessions:
                return []
            
            session = self.sessions[client_id]
            messages = session.messages
            
            # Limit messages if requested
            if max_messages:
                messages = messages[-max_messages:]
            
            # Convert to dict format
            history = []
            for msg in messages:
                msg_dict = {
                    "message_id": msg.message_id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                }
                
                if include_metadata:
                    msg_dict["metadata"] = msg.metadata
                
                history.append(msg_dict)
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error getting chat history for {client_id}: {e}")
            return []
    
    def get_conversation_context(self, client_id: str, max_messages: int = 10) -> str:
        """Get conversation context as a formatted string."""
        try:
            if client_id not in self.sessions:
                return ""
            
            session = self.sessions[client_id]
            messages = session.messages[-max_messages:] if max_messages else session.messages
            
            context_parts = []
            for msg in messages:
                role_label = "User" if msg.role == "user" else "Assistant"
                context_parts.append(f"{role_label}: {msg.content}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            self.logger.error(f"Error getting conversation context for {client_id}: {e}")
            return ""
    
    def clear_history(self, client_id: str) -> bool:
        """Clear chat history for a client."""
        try:
            if client_id not in self.sessions:
                return False
            
            session = self.sessions[client_id]
            session.messages = []
            session.context = {}
            session.last_activity = datetime.now().isoformat()
            
            self.logger.info(f"Cleared chat history for client: {client_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing history for {client_id}: {e}")
            return False
    
    def delete_session(self, client_id: str) -> bool:
        """Delete a client's session."""
        try:
            if client_id not in self.sessions:
                return False
            
            del self.sessions[client_id]
            self.active_sessions -= 1
            
            self.logger.info(f"Deleted session for client: {client_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting session for {client_id}: {e}")
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.session_timeout_hours)
            expired_clients = []
            
            for client_id, session in self.sessions.items():
                last_activity = datetime.fromisoformat(session.last_activity)
                if last_activity < cutoff_time:
                    expired_clients.append(client_id)
            
            # Remove expired sessions
            for client_id in expired_clients:
                del self.sessions[client_id]
                self.active_sessions -= 1
            
            if expired_clients:
                self.logger.info(f"Cleaned up {len(expired_clients)} expired sessions")
            
            return len(expired_clients)
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired sessions: {e}")
            return 0
    
    def get_session_stats(self, client_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session."""
        try:
            if client_id not in self.sessions:
                return {"error": "Session not found"}
            
            session = self.sessions[client_id]
            
            # Count messages by role
            user_messages = sum(1 for msg in session.messages if msg.role == "user")
            assistant_messages = sum(1 for msg in session.messages if msg.role == "assistant")
            
            return {
                "session_id": session.session_id,
                "client_id": client_id,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "total_messages": len(session.messages),
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "max_history": session.max_history,
                "context_keys": list(session.context.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session stats for {client_id}: {e}")
            return {"error": str(e)}
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics."""
        try:
            # Calculate session ages
            now = datetime.now()
            session_ages = []
            for session in self.sessions.values():
                created = datetime.fromisoformat(session.created_at)
                age_hours = (now - created).total_seconds() / 3600
                session_ages.append(age_hours)
            
            return {
                "total_sessions": len(self.sessions),
                "active_sessions": self.active_sessions,
                "total_messages": self.total_messages,
                "average_session_age_hours": sum(session_ages) / len(session_ages) if session_ages else 0,
                "oldest_session_hours": max(session_ages) if session_ages else 0,
                "newest_session_hours": min(session_ages) if session_ages else 0,
                "max_sessions": self.max_sessions,
                "session_timeout_hours": self.session_timeout_hours
            }
            
        except Exception as e:
            self.logger.error(f"Error getting global stats: {e}")
            return {"error": str(e)}
    
    def export_session(self, client_id: str) -> Dict[str, Any]:
        """Export a session for backup/analysis."""
        try:
            if client_id not in self.sessions:
                return {"error": "Session not found"}
            
            session = self.sessions[client_id]
            return asdict(session)
            
        except Exception as e:
            self.logger.error(f"Error exporting session {client_id}: {e}")
            return {"error": str(e)}
    
    def import_session(self, session_data: Dict[str, Any]) -> bool:
        """Import a session from backup."""
        try:
            client_id = session_data.get("client_id")
            if not client_id:
                return False
            
            # Create session from data
            session = ChatSession(
                session_id=session_data.get("session_id", str(uuid.uuid4())),
                client_id=client_id,
                created_at=session_data.get("created_at", datetime.now().isoformat()),
                last_activity=session_data.get("last_activity", datetime.now().isoformat()),
                messages=[ChatMessage(**msg) for msg in session_data.get("messages", [])],
                context=session_data.get("context", {}),
                max_history=session_data.get("max_history", 50)
            )
            
            self.sessions[client_id] = session
            self.active_sessions += 1
            
            self.logger.info(f"Imported session for client: {client_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing session: {e}")
            return False


# Global chat history manager instance
chat_manager = ChatHistoryManager()
