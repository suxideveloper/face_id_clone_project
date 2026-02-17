import json
import os
from app.core.config import settings

class UserDatabase:
    def __init__(self):
        self.db_path = os.path.join(settings.DATA_DIR, "users.json")
        self.users = {}
        self.load()
    
    def load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    self.users = json.load(f)
            except Exception as e:
                print(f"Error loading user database: {e}")
                self.users = {}
        else:
            self.users = {}
    
    def save(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def create_user(self, user_id: str, data: dict) -> bool:
        """Create a new user with profile data."""
        self.users[user_id] = {
            "full_name": data.get("full_name", ""),
            "phone": data.get("phone", ""),
            "department": data.get("department", ""),
            "position": data.get("position", ""),
            "registered": True
        }
        self.save()
        return True
    
    def get_user(self, user_id: str) -> dict:
        """Get user data by ID (name)."""
        return self.users.get(user_id, None)
    
    def get_all_users(self) -> dict:
        """Get all users."""
        return self.users
    
    def update_user(self, user_id: str, data: dict) -> bool:
        """Update user profile data."""
        if user_id not in self.users:
            return False
        self.users[user_id].update(data)
        self.save()
        return True
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        if user_id in self.users:
            del self.users[user_id]
            self.save()
            return True
        return False
    
    def rename_user(self, old_id: str, new_id: str) -> bool:
        """Rename a user (change their ID/name)."""
        if old_id in self.users and new_id not in self.users:
            self.users[new_id] = self.users.pop(old_id)
            self.save()
            return True
        return False

user_db = UserDatabase()
