import sys
from app.services import admin_db

def create_kiosk_user():
    print("=== Create Kiosk User ===")
    username = input("Username (default: kiosk): ").strip() or "kiosk"
    password = input("Password: ").strip()
    
    if not password:
        print("Error: Password is required")
        return

    success, message = admin_db.create_admin(username, password, role="kiosk")
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ Error: {message}")

if __name__ == "__main__":
    create_kiosk_user()
