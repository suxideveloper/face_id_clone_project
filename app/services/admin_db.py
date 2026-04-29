import hashlib
import secrets

from app.core.database import SessionLocal
from app.core.models import Admin


def _get_password_hash(password: str, salt: str) -> str:
    """Create a secure hash of the password using a salt."""
    return hashlib.sha256((password + salt).encode('utf-8')).hexdigest()


def _load_admins():
    """Load all admins as a dict (for backward compatibility with routes.py)."""
    with SessionLocal() as db:
        rows = db.query(Admin).all()
        return {
            a.username: {
                "username": a.username,
                "hashed_password": a.password_hash.split(":")[1] if ":" in a.password_hash else a.password_hash,
                "salt": a.password_hash.split(":")[0] if ":" in a.password_hash else "",
                "role": a.role or "admin",
            }
            for a in rows
        }


def create_admin(username, password, role="admin"):
    with SessionLocal() as db:
        if db.query(Admin).filter(Admin.username == username).first():
            return False, "User already exists"

        salt = secrets.token_hex(16)
        hashed_pw = _get_password_hash(password, salt)

        admin = Admin(
            username=username,
            password_hash=f"{salt}:{hashed_pw}",
            role=role,
        )
        db.add(admin)
        db.commit()
        return True, f"User '{username}' with role '{role}' created successfully"


def verify_admin(username, password):
    with SessionLocal() as db:
        admin = db.query(Admin).filter(Admin.username == username).first()
        if not admin:
            return None

        stored = admin.password_hash
        # Support both new "salt:hash" format and legacy format
        if ":" in stored:
            salt, hashed_pw = stored.split(":", 1)
        else:
            # Legacy: salt was stored separately in JSON; try without salt
            salt = ""
            hashed_pw = stored

        if _get_password_hash(password, salt) == hashed_pw:
            return {
                "username": admin.username,
                "role": admin.role or "admin",
            }
        return None
