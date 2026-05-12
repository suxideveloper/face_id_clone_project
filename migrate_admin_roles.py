"""
Migratsiya skripti: Mavjud barcha 'admin' rollarni 'superadmin' ga o'tkazadi.

Yangi tizimda:
  superadmin = to'liq ruxsat (xodim qo'shish, o'chirish, sozlamalar)
  admin      = faqat ko'rish (hisobotlar, davomat, xodimlar ro'yxati)

Ishlatish:
  python migrate_admin_roles.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.admin_db import migrate_existing_admins_to_superadmin, list_all_admins

def run():
    print("=" * 50)
    print("Admin Rol Migratsiyasi")
    print("=" * 50)

    # Migratsiyadan oldin ko'rsatish
    admins_before = list_all_admins()
    print(f"\nMigratsiyadan OLDIN ({len(admins_before)} admin):")
    for a in admins_before:
        print(f"  - {a['username']:20s}  rol: {a['role']}")

    confirm = input("\nBarcha 'admin' rollarni 'superadmin' ga o'tkazasizmi? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Migratsiya bekor qilindi.")
        return

    count = migrate_existing_admins_to_superadmin()
    print(f"\n✅ {count} ta admin 'superadmin' ga o'tkazildi.")

    # Migratsiyadan keyin ko'rsatish
    admins_after = list_all_admins()
    print(f"\nMigratsiyadan KEYIN ({len(admins_after)} admin):")
    for a in admins_after:
        print(f"  - {a['username']:20s}  rol: {a['role']}")

    print("\nEndi yangi 'admin' (faqat ko'rish) yaratish uchun:")
    print("  python create_superuser.py  — superadmin yaratish")
    print("  python create_admin_user.py — admin (read-only) yaratish")

if __name__ == "__main__":
    run()
