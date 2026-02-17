import getpass
from app.services import admin_db

def create_superuser():
    print("=== Create Superuser ===")
    username = input("Username: ").strip()
    if not username:
        print("Error: Username cannot be empty.")
        return

    password = getpass.getpass("Password: ")
    confirm_password = getpass.getpass("Confirm Password: ")

    if password != confirm_password:
        print("Error: Passwords do not match.")
        return

    if not password:
        print("Error: Password cannot be empty.")
        return

    success, message = admin_db.create_admin(username, password)
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")

if __name__ == "__main__":
    create_superuser()
