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


def list_all_admins():
    """Barcha admin va superadminlar ro'yxatini qaytaradi."""
    with SessionLocal() as db:
        rows = db.query(Admin).all()
        return [
            {
                "username": a.username,
                "role": a.role or "admin",
            }
            for a in rows
            if a.role not in ("kiosk",)  # kiosk userlarni chiqarib tashlash
        ]


def list_all_kiosk_users():
    """Barcha kiosk userlarni qaytaradi."""
    with SessionLocal() as db:
        rows = db.query(Admin).filter(Admin.role == "kiosk").all()
        return [{"username": a.username, "role": a.role} for a in rows]


def delete_admin(username):
    """Admin yoki superadminni o'chiradi. kiosk rolini o'chirib bo'lmaydi bu funksiya orqali."""
    with SessionLocal() as db:
        admin = db.query(Admin).filter(Admin.username == username).first()
        if not admin:
            return False, "User not found"
        if admin.role == "kiosk":
            return False, "Kiosk users cannot be deleted via this function"
        db.delete(admin)
        db.commit()
        return True, f"User '{username}' deleted successfully"


def change_admin_password(username, new_password):
    """Admin parolini o'zgartiradi."""
    with SessionLocal() as db:
        admin = db.query(Admin).filter(Admin.username == username).first()
        if not admin:
            return False, "User not found"

        salt = secrets.token_hex(16)
        hashed_pw = _get_password_hash(new_password, salt)
        admin.password_hash = f"{salt}:{hashed_pw}"
        db.commit()
        return True, f"Password for '{username}' changed successfully"


def change_admin_role(username, new_role):
    """Admin rolini o'zgartiradi."""
    allowed_roles = ("admin", "superadmin", "kiosk")
    if new_role not in allowed_roles:
        return False, f"Invalid role. Allowed: {allowed_roles}"

    with SessionLocal() as db:
        admin = db.query(Admin).filter(Admin.username == username).first()
        if not admin:
            return False, "User not found"
        old_role = admin.role
        admin.role = new_role
        db.commit()
        return True, f"Role for '{username}' changed from '{old_role}' to '{new_role}'"


def migrate_existing_admins_to_superadmin():
    """Mavjud barcha 'admin' rollarini 'superadmin' ga o'tkazadi."""
    with SessionLocal() as db:
        rows = db.query(Admin).filter(Admin.role == "admin").all()
        count = 0
        for a in rows:
            a.role = "superadmin"
            count += 1
        db.commit()
        return count
