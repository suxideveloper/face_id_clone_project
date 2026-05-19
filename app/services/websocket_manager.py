from fastapi import WebSocket
from typing import List
import json
import asyncio

class WebSocketManager:
    """
    Manages WebSocket connections and broadcasts attendance events to all connected clients.
    """
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        print(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        print(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return
        
        message_json = json.dumps(message)
        disconnected = []
        
        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_text(message_json)
                except Exception as e:
                    print(f"Error sending to WebSocket: {e}")
                    disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            await self.disconnect(conn)
    
    async def send_check_in(self, name: str, full_name: str = None):
        """Send a check-in event to all clients."""
        display_name = full_name if full_name else name
        await self.broadcast({
            "status": "check_in",
            "name": display_name,
            "worker_id": name,
            "message": "Welcome!",
            "timestamp": self._get_timestamp()
        })
    
    async def send_check_out(self, name: str, full_name: str = None, working_time: str = None, check_in_time: str = None):
        """Send a check-out event to all clients."""
        display_name = full_name if full_name else name
        await self.broadcast({
            "status": "check_out",
            "name": display_name,
            "worker_id": name,
            "message": "Time Updated!",
            "working_time": working_time,
            "check_in_time": check_in_time,
            "timestamp": self._get_timestamp()
        })

    async def send_not_employee(self):
        """Send a 'not_employee' event: person passed liveness but is NOT registered."""
        await self.broadcast({
            "status": "not_employee",
            "name": "Unknown",
            "worker_id": "Unknown",
            "message": "Bu shaxs tizimda ro'yxatdan o'tmagan",
            "timestamp": self._get_timestamp()
        })
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")


# Singleton instance
ws_manager = WebSocketManager()
