import sys
import os

# Loyiha asosiy papkasini yo'lga qo'shish
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.core.database import engine

def run():
    print("=" * 50)
    print("🗑 Barcha admin va superadminlarni o'chirish")
    print("=" * 50)
    
    confirm = input("\nDIQQAT: Haqiqatan ham BARCHA adminlarni o'chirib tashlamoqchimisiz? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Jarayon bekor qilindi.")
        return

    with engine.connect() as conn:
        try:
            result = conn.execute(text("DELETE FROM admins"))
            conn.commit()
            print(f"\n✅ Muvaffaqiyatli o'chirildi! (O'chirilgan adminlar soni: {result.rowcount} ta)")
            print("\nYangi superadmin yaratish uchun quyidagi buyruqni ishlating:")
            print("  python create_superuser.py")
        except Exception as e:
            print(f"❌ Xatolik yuz berdi: {e}")
            conn.rollback()

if __name__ == "__main__":
    run()
