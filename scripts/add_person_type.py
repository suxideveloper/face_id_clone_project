"""
Migration: users jadvaliga person_type va notes ustunlarini qo'shish.
Mavjud is_doctorant qiymatlari asosida person_type avtomatik to'ldiriladi.

Ishlatish:
    python scripts/add_person_type.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.core.database import SessionLocal, engine

def run_migration():
    print("🚀 Migration boshlandi: person_type va notes ustunlari qo'shilmoqda...")

    with engine.connect() as conn:
        # 1. person_type ustunini qo'shish (agar mavjud bo'lmasa)
        try:
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN person_type VARCHAR DEFAULT 'staff' NOT NULL"
            ))
            conn.commit()
            print("✅ person_type ustuni qo'shildi.")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
                print("ℹ️  person_type ustuni allaqachon mavjud, o'tkazib yuborildi.")
                conn.rollback()
            else:
                print(f"❌ person_type qo'shishda xato: {e}")
                conn.rollback()

        # 2. notes ustunini qo'shish (agar mavjud bo'lmasa)
        try:
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN notes TEXT DEFAULT ''"
            ))
            conn.commit()
            print("✅ notes ustuni qo'shildi.")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
                print("ℹ️  notes ustuni allaqachon mavjud, o'tkazib yuborildi.")
                conn.rollback()
            else:
                print(f"❌ notes qo'shishda xato: {e}")
                conn.rollback()

        # 3. Mavjud doktorantlarni person_type='doctorant' ga o'tkazish
        try:
            result = conn.execute(text(
                "UPDATE users SET person_type = 'doctorant' WHERE is_doctorant = TRUE AND (person_type IS NULL OR person_type = 'staff')"
            ))
            conn.commit()
            print(f"✅ {result.rowcount} ta doktorant person_type='doctorant' ga o'tkazildi.")
        except Exception as e:
            print(f"❌ Doktorant migratsiyasida xato: {e}")
            conn.rollback()

        # 4. NULL person_type larni 'staff' ga o'tkazish
        try:
            result = conn.execute(text(
                "UPDATE users SET person_type = 'staff' WHERE person_type IS NULL"
            ))
            conn.commit()
            print(f"✅ {result.rowcount} ta NULL person_type 'staff' ga o'tkazildi.")
        except Exception as e:
            print(f"❌ NULL migration xato: {e}")
            conn.rollback()

        # 5. NULL notes larni '' ga o'tkazish
        try:
            conn.execute(text(
                "UPDATE users SET notes = '' WHERE notes IS NULL"
            ))
            conn.commit()
            print("✅ NULL notes qiymatlari tozalandi.")
        except Exception as e:
            print(f"⚠️  Notes NULL migration: {e}")
            conn.rollback()

    # 6. Natijani tekshirish
    print("\n📊 Migration natijasi:")
    with SessionLocal() as db:
        from sqlalchemy import text as t
        rows = db.execute(t(
            "SELECT person_type, COUNT(*) as cnt FROM users GROUP BY person_type ORDER BY cnt DESC"
        )).fetchall()
        for row in rows:
            print(f"   {row[0] or 'NULL':20s} → {row[1]} ta foydalanuvchi")

    print("\n✅ Migration muvaffaqiyatli yakunlandi!")

if __name__ == "__main__":
    run_migration()
