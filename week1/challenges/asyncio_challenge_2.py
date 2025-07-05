#!/usr/bin/env python3
"""
Asyncio Challenge 2: Real-Time Chat Server

Implement a real-time chat server using asyncio with WebSocket connections,
supporting multiple chat rooms, user management, and message broadcasting.

Requirements:
1. Handle multiple concurrent WebSocket connections
2. Support multiple chat rooms with user management
3. Implement message broadcasting and private messaging
4. Add advanced features like typing indicators and file sharing
5. Include administration commands and real-time statistics

Time: 50-65 minutes
"""

import asyncio
import websockets
import json
import time
import signal
import sys
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Any
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import hashlib
import base64
import uuid
from rich.console import Console
from rich.table import Table
from rich.live import Live

console = Console()

class MessageType(Enum):
    """Types of chat messages"""
    CHAT = "chat"
    JOIN = "join"
    LEAVE = "leave"
    PRIVATE = "private"
    TYPING = "typing"
    FILE = "file"
    ADMIN = "admin"
    SYSTEM = "system"

class UserStatus(Enum):
    """User status types"""
    ONLINE = "online"
    AWAY = "away"
    BUSY = "busy"
    OFFLINE = "offline"

@dataclass
class User:
    """Represents a chat user"""
    id: str
    username: str
    websocket: websockets.WebSocketServerProtocol
    current_room: str = "general"
    status: UserStatus = UserStatus.ONLINE
    is_admin: bool = False
    is_muted: bool = False
    join_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    typing_in_room: Optional[str] = None

@dataclass
class ChatMessage:
    """Represents a chat message"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.CHAT
    sender_id: str = ""
    sender_username: str = ""
    content: str = ""
    room: str = ""
    timestamp: float = field(default_factory=time.time)
    private_to: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class ChatRoom:
    """Represents a chat room"""
    name: str
    description: str = ""
    created_by: str = ""
    created_at: float = field(default_factory=time.time)
    users: Set[str] = field(default_factory=set)
    message_history: deque = field(default_factory=lambda: deque(maxlen=100))
    is_private: bool = False
    max_users: Optional[int] = None

class MessageHistory:
    """Manages chat message history and persistence"""

    def __init__(self, max_messages_per_room: int = 1000):
        self.max_messages_per_room = max_messages_per_room
        self.room_histories = defaultdict(lambda: deque(maxlen=max_messages_per_room))

    async def add_message(self, message: ChatMessage):
        """Add message to history"""
        # TODO: Implement message history storage:
        # - Add to room history
        # - Optionally persist to file/database
        # - Maintain size limits
        pass

    async def get_room_history(self, room: str, limit: int = 50) -> List[ChatMessage]:
        """Get recent messages for a room"""
        # TODO: Implement history retrieval
        pass

    async def search_messages(self, query: str, room: str = None) -> List[ChatMessage]:
        """Search messages by content"""
        # TODO: Implement message search
        pass

class UserManager:
    """Manages user connections and authentication"""

    def __init__(self):
        self.users: Dict[str, User] = {}
        self.username_to_id: Dict[str, str] = {}
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))

    async def authenticate_user(self, websocket, auth_data: Dict) -> Optional[User]:
        """Authenticate a user connection"""
        # TODO: Implement user authentication:
        # - Validate username/password
        # - Check for existing connections
        # - Create User object
        # - Handle reconnections
        pass

    async def add_user(self, user: User):
        """Add a user to the manager"""
        # TODO: Implement user addition:
        # - Add to users dict
        # - Update username mapping
        # - Initialize rate limiting
        pass

    async def remove_user(self, user_id: str):
        """Remove a user from the manager"""
        # TODO: Implement user removal:
        # - Remove from all rooms
        # - Clean up connections
        # - Notify other users
        pass

    async def is_rate_limited(self, user_id: str) -> bool:
        """Check if user is rate limited"""
        # TODO: Implement rate limiting:
        # - Track message timestamps
        # - Check against rate limit thresholds
        # - Clean old timestamps
        pass

    async def get_user_stats(self) -> Dict:
        """Get user statistics"""
        # TODO: Compile user statistics
        pass

class RoomManager:
    """Manages chat rooms and membership"""

    def __init__(self):
        self.rooms: Dict[str, ChatRoom] = {}
        self._create_default_rooms()

    def _create_default_rooms(self):
        """Create default chat rooms"""
        # TODO: Create default rooms like "general", "tech-talk", etc.
        pass

    async def create_room(self, name: str, creator_id: str,
                         description: str = "", is_private: bool = False,
                         max_users: Optional[int] = None) -> ChatRoom:
        """Create a new chat room"""
        # TODO: Implement room creation:
        # - Validate room name
        # - Create ChatRoom object
        # - Add to rooms dict
        # - Set permissions
        pass

    async def join_room(self, user_id: str, room_name: str) -> bool:
        """Add user to a room"""
        # TODO: Implement room joining:
        # - Check room exists
        # - Check permissions and capacity
        # - Add user to room
        # - Notify other users
        pass

    async def leave_room(self, user_id: str, room_name: str) -> bool:
        """Remove user from a room"""
        # TODO: Implement room leaving:
        # - Remove user from room
        # - Notify other users
        # - Handle empty rooms
        pass

    async def delete_room(self, room_name: str, user_id: str) -> bool:
        """Delete a room (admin only)"""
        # TODO: Implement room deletion
        pass

    async def get_room_list(self, user_id: str) -> List[Dict]:
        """Get list of available rooms for user"""
        # TODO: Return room list with user counts and descriptions
        pass

class FileManager:
    """Manages file uploads and sharing"""

    def __init__(self, upload_dir: str = "uploads", max_file_size: int = 10*1024*1024):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.max_file_size = max_file_size
        self.allowed_extensions = {'.txt', '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.doc', '.docx'}

    async def handle_file_upload(self, user_id: str, file_data: bytes,
                                filename: str, room: str) -> Optional[str]:
        """Handle file upload"""
        # TODO: Implement file upload:
        # - Validate file size and type
        # - Generate unique filename
        # - Save file securely
        # - Return file URL/ID
        pass

    async def get_file_info(self, file_id: str) -> Optional[Dict]:
        """Get file information"""
        # TODO: Return file metadata
        pass

class ChatServer:
    """Main chat server implementation"""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.user_manager = UserManager()
        self.room_manager = RoomManager()
        self.message_history = MessageHistory()
        self.file_manager = FileManager()
        self.server = None
        self.shutdown_event = asyncio.Event()
        self.statistics = {
            'start_time': time.time(),
            'total_connections': 0,
            'total_messages': 0,
            'peak_concurrent_users': 0
        }

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        console.print("\n[yellow]Received shutdown signal. Stopping server...[/yellow]")
        asyncio.create_task(self.shutdown())

    async def handle_client(self, websocket, path):
        """Handle a new client connection"""
        # TODO: Implement client connection handling:
        # - Authenticate user
        # - Register connection
        # - Handle messages
        # - Clean up on disconnect
        pass

    async def handle_message(self, user: User, raw_message: str):
        """Handle incoming message from user"""
        # TODO: Implement message handling:
        # - Parse JSON message
        # - Validate message type and content
        # - Check rate limiting
        # - Route to appropriate handler
        # - Update user activity
        pass

    async def handle_chat_message(self, user: User, message_data: Dict):
        """Handle regular chat message"""
        # TODO: Implement chat message handling:
        # - Create ChatMessage object
        # - Validate content and room
        # - Broadcast to room members
        # - Add to message history
        pass

    async def handle_private_message(self, user: User, message_data: Dict):
        """Handle private message between users"""
        # TODO: Implement private messaging:
        # - Validate recipient
        # - Send to specific user
        # - Add to private message history
        pass

    async def handle_typing_indicator(self, user: User, message_data: Dict):
        """Handle typing indicator"""
        # TODO: Implement typing indicators:
        # - Track typing state
        # - Broadcast to room members
        # - Auto-clear after timeout
        pass

    async def handle_admin_command(self, user: User, message_data: Dict):
        """Handle admin commands"""
        # TODO: Implement admin commands:
        # - Kick/ban users
        # - Mute users
        # - Create/delete rooms
        # - View server statistics
        pass

    async def broadcast_to_room(self, room: str, message: ChatMessage,
                              exclude_user: Optional[str] = None):
        """Broadcast message to all users in a room"""
        # TODO: Implement room broadcasting:
        # - Get all users in room
        # - Send message to each user's websocket
        # - Handle connection errors
        pass

    async def send_to_user(self, user_id: str, message: Dict) -> bool:
        """Send message to specific user"""
        # TODO: Implement direct user messaging:
        # - Find user websocket
        # - Send JSON message
        # - Handle connection errors
        pass

    async def start_server(self):
        """Start the chat server"""
        # TODO: Implement server startup:
        # - Start WebSocket server
        # - Start monitoring tasks
        # - Display server information
        pass

    async def shutdown(self):
        """Graceful server shutdown"""
        # TODO: Implement graceful shutdown:
        # - Notify all users
        # - Close all connections
        # - Stop server
        # - Display final statistics
        pass

    async def _monitor_connections(self):
        """Monitor and display server statistics"""
        # TODO: Implement connection monitoring:
        # - Track active connections
        # - Update statistics
        # - Display real-time dashboard
        pass

    async def _cleanup_inactive_users(self):
        """Clean up inactive users and expired typing indicators"""
        # TODO: Implement cleanup tasks:
        # - Remove inactive users
        # - Clear expired typing indicators
        # - Update user presence
        pass

async def create_test_client():
    """Create a test client for demonstration"""
    # TODO: Implement simple test client:
    # - Connect to server
    # - Send test messages
    # - Demonstrate features
    pass

async def main():
    """Start the chat server"""

    try:
        console.print("[bold green]Real-Time Chat Server[/bold green]")
        console.print("WebSocket server starting on ws://localhost:8765")
        console.print("Features: Multiple rooms, Private messaging, File sharing, Admin commands")
        console.print("Press Ctrl+C to stop\n")

        # Create and start server
        server = ChatServer(host="localhost", port=8765)
        await server.start_server()

        # Keep server running
        try:
            await server.shutdown_event.wait()
        except KeyboardInterrupt:
            pass

        # Shutdown
        await server.shutdown()

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

# TODO: Implementation checklist
"""
□ Implement User and ChatRoom management
□ Implement WebSocket connection handling
□ Implement message routing and broadcasting
□ Implement private messaging system
□ Implement typing indicators with auto-cleanup
□ Implement file upload and sharing
□ Implement admin commands and permissions
□ Implement rate limiting and security measures
□ Implement message history and persistence
□ Implement real-time monitoring dashboard
□ Add comprehensive error handling
□ Test with multiple concurrent connections
□ Optimize for 1000+ concurrent users
"""