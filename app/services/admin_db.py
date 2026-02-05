import json
import os
import hashlib
import secrets

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
ADMIN_FILE = os.path.join(DATA_DIR, "admins.json")

def _get_password_hash(password: str, salt: str) -> str:
    """Create a secure hash of the password using a salt."""
    return hashlib.sha256((password + salt).encode('utf-8')).hexdigest()

def _load_admins():
    if not os.path.exists(ADMIN_FILE):
        return {}
    try:
        with open(ADMIN_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def _save_admins(admins):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ADMIN_FILE, 'w') as f:
        json.dump(admins, f, indent=4)

def create_admin(username, password):
    admins = _load_admins()
    if username in admins:
        return False, "User already exists"
    
    salt = secrets.token_hex(16)
    hashed_pw = _get_password_hash(password, salt)
    
    admins[username] = {
        "username": username,
        "hashed_password": hashed_pw,
        "salt": salt
    }
    _save_admins(admins)
    return True, "Admin created successfully"

def verify_admin(username, password):
    admins = _load_admins()
    if username not in admins:
        return False
    
    admin = admins[username]
    salt = admin.get("salt", "")
    hashed_pw = _get_password_hash(password, salt)
    
    if hashed_pw == admin["hashed_password"]:
        return True
    return False
