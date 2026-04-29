from app.core.database import SessionLocal
from app.core.models import User

class UserDatabase:
    def create_user(self, user_id: str, data: dict) -> bool:
        """Create a new user with profile data."""
        with SessionLocal() as db:
            if db.query(User).filter(User.username == user_id).first():
                return False
            u = User(
                username=user_id,
                full_name=data.get("full_name", ""),
                phone=data.get("phone", ""),
                department=data.get("department", ""),
                position=data.get("position", ""),
                registered=True
            )
            db.add(u)
            db.commit()
            return True
    
    def get_user(self, user_id: str) -> dict:
        """Get user data by ID (name)."""
        with SessionLocal() as db:
            u = db.query(User).filter(User.username == user_id).first()
            if u:
                return {
                    "full_name": u.full_name,
                    "phone": u.phone,
                    "department": u.department,
                    "position": u.position,
                    "registered": u.registered
                }
            return None
    
    def get_all_users(self) -> dict:
        """Get all users."""
        with SessionLocal() as db:
            users = db.query(User).all()
            return {
                u.username: {
                    "full_name": u.full_name,
                    "phone": u.phone,
                    "department": u.department,
                    "position": u.position,
                    "registered": u.registered
                } for u in users
            }
    
    def update_user(self, user_id: str, data: dict) -> bool:
        """Update user profile data."""
        with SessionLocal() as db:
            u = db.query(User).filter(User.username == user_id).first()
            if not u:
                return False
            for k, v in data.items():
                if hasattr(u, k):
                    setattr(u, k, v)
            db.commit()
            return True
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        with SessionLocal() as db:
            u = db.query(User).filter(User.username == user_id).first()
            if u:
                db.delete(u)
                db.commit()
                return True
            return False
    
    def rename_user(self, old_id: str, new_id: str) -> bool:
        """Rename a user (change their ID/name)."""
        with SessionLocal() as db:
            u = db.query(User).filter(User.username == old_id).first()
            if u:
                u.username = new_id
                db.commit()
                return True
            return False

user_db = UserDatabase()
