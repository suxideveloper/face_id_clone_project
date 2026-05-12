"""
Yangi admin (faqat ko'rish huquqi) yaratish skripti.

admin roli imkoniyatlari:
  ✅ Dashboard, hisobotlar, davomat, xodimlar ro'yxatini ko'rish
  ✅ O'z parolini o'zgartirish
  ❌ Xodim qo'shish, tahrirlash, o'chirish
  ❌ Sozlamalar, bo'limlar, Telegram boshqaruvi
  ❌ Boshqa adminlarni boshqarish

To'liq ruxsat uchun: python create_superuser.py
"""
import getpass
from app.services import admin_db

def create_admin_user():
    print("=== Yangi Admin Yaratish (faqat ko'rish) ===")
    username = input("Username: ").strip()
    if not username:
        print("Xato: Username bo'sh bo'lishi mumkin emas.")
        return

    password = getpass.getpass("Parol: ")
    confirm_password = getpass.getpass("Parolni tasdiqlang: ")

    if password != confirm_password:
        print("Xato: Parollar mos kelmadi.")
        return
    if not password:
        print("Xato: Parol bo'sh bo'lishi mumkin emas.")
        return
    if len(password) < 6:
        print("Xato: Parol kamida 6 ta belgi bo'lishi kerak.")
        return

    success, message = admin_db.create_admin(username, password, role="admin")
    if success:
        print(f"✅ {message}")
        print(f"   Rol: admin (faqat ko'rish huquqi)")
    else:
        print(f"❌ {message}")

if __name__ == "__main__":
    create_admin_user()
