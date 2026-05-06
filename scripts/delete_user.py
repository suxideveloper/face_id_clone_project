"""
Foydalanuvchini tizimdan to'liq o'chirish skripti.
Ishlatish: python scripts/delete_user.py <user_id>

Masalan:
    python scripts/delete_user.py Murod
"""

import sys
import os
import pickle
import shutil

# Loyiha papkasini PATH ga qo'shamiz
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app.core.database import SessionLocal
from app.core.models import User, Attendance, FaceEncoding

DATA_DIR = os.path.join(BASE_DIR, "data")
ENCODINGS_FILE = os.path.join(DATA_DIR, "encodings.pkl")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
SNAPSHOTS_DIR = os.path.join(DATA_DIR, "attendance_snapshots")


def delete_user(user_id: str):
    deleted = []
    not_found = []

    print(f"\n{'='*50}")
    print(f"  Foydalanuvchi o'chirilmoqda: '{user_id}'")
    print(f"{'='*50}")

    # 1. PostgreSQL DB dan o'chirish
    print("\n[1/4] PostgreSQL ma'lumotlar bazasi...")
    try:
        with SessionLocal() as db:
            user = db.query(User).filter(User.username == user_id).first()
            if user:
                # cascade="all, delete-orphan" - Attendance va FaceEncoding avtomatik o'chadi
                att_count = db.query(Attendance).filter(Attendance.username == user_id).count()
                enc_count = db.query(FaceEncoding).filter(FaceEncoding.username == user_id).count()
                db.delete(user)
                db.commit()
                deleted.append(f"DB: foydalanuvchi + {att_count} davomat + {enc_count} encoding")
                print(f"  ✅ Foydalanuvchi, {att_count} davomat va {enc_count} encoding o'chirildi")
            else:
                not_found.append("DB: foydalanuvchi topilmadi")
                print(f"  ⚠️  Foydalanuvchi DB da topilmadi")
    except Exception as e:
        print(f"  ❌ DB xatosi: {e}")

    # 2. encodings.pkl dan yuz ma'lumotlarini o'chirish
    print("\n[2/4] Yuz encodings fayli (encodings.pkl)...")
    try:
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)

            # encodings.pkl formatini aniqlash
            if isinstance(data, dict):
                # Format: {"names": [...], "encodings": [...]}
                if "names" in data and "encodings" in data:
                    names = data["names"]
                    encodings = data["encodings"]
                    before = len(names)
                    filtered = [(n, e) for n, e in zip(names, encodings) if n != user_id]
                    if filtered:
                        new_names, new_encodings = zip(*filtered)
                    else:
                        new_names, new_encodings = [], []
                    data["names"] = list(new_names)
                    data["encodings"] = list(new_encodings)
                    removed = before - len(new_names)
                # Format: {user_id: [embedding1, embedding2, ...]}
                elif user_id in data:
                    removed = len(data[user_id])
                    del data[user_id]
                else:
                    removed = 0

            with open(ENCODINGS_FILE, "wb") as f:
                pickle.dump(data, f)

            if removed > 0:
                deleted.append(f"encodings.pkl: {removed} ta yuz ma'lumoti")
                print(f"  ✅ {removed} ta yuz encoding o'chirildi")
            else:
                not_found.append("encodings.pkl: bu foydalanuvchi uchun encoding topilmadi")
                print(f"  ⚠️  Bu foydalanuvchi uchun encoding topilmadi")
        else:
            print(f"  ⚠️  encodings.pkl fayli topilmadi")
    except Exception as e:
        print(f"  ❌ Encoding xatosi: {e}")

    # 3. Rasmlar papkasini o'chirish (data/images/<user_id>/)
    print("\n[3/4] Yuz rasmlari papkasi...")
    user_images_dir = os.path.join(IMAGES_DIR, user_id)
    try:
        if os.path.exists(user_images_dir):
            count = len([f for f in os.listdir(user_images_dir) if os.path.isfile(os.path.join(user_images_dir, f))])
            shutil.rmtree(user_images_dir)
            deleted.append(f"images/{user_id}/: {count} ta rasm")
            print(f"  ✅ {count} ta rasm o'chirildi ({user_images_dir})")
        else:
            not_found.append(f"images/{user_id}/: papka topilmadi")
            print(f"  ⚠️  Rasmlar papkasi topilmadi: {user_images_dir}")
    except Exception as e:
        print(f"  ❌ Rasmlar xatosi: {e}")

    # 4. Davomat snapshotlarini o'chirish
    print("\n[4/4] Davomat snapshot rasmlari...")
    try:
        if os.path.exists(SNAPSHOTS_DIR):
            removed_snaps = 0
            for fname in os.listdir(SNAPSHOTS_DIR):
                if fname.startswith(f"{user_id}_") or fname.startswith(f"{user_id}."):
                    os.remove(os.path.join(SNAPSHOTS_DIR, fname))
                    removed_snaps += 1
            if removed_snaps > 0:
                deleted.append(f"snapshots: {removed_snaps} ta rasm")
                print(f"  ✅ {removed_snaps} ta snapshot o'chirildi")
            else:
                print(f"  ℹ️  Snapshot topilmadi (bu normal)")
        else:
            print(f"  ℹ️  Snapshots papkasi yo'q")
    except Exception as e:
        print(f"  ❌ Snapshot xatosi: {e}")

    # Xulosa
    print(f"\n{'='*50}")
    print(f"  XULOSA")
    print(f"{'='*50}")
    if deleted:
        print(f"\n  ✅ O'chirildi:")
        for item in deleted:
            print(f"     • {item}")
    if not_found:
        print(f"\n  ⚠️  Topilmadi (allaqachon yo'q bo'lishi mumkin):")
        for item in not_found:
            print(f"     • {item}")
    print(f"\n  Foydalanuvchi '{user_id}' tizimdan to'liq o'chirildi ✅\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nIshlatish: python scripts/delete_user.py <user_id>")
        print("Misol:     python scripts/delete_user.py Murod\n")
        sys.exit(1)

    user_id = sys.argv[1]

    confirm = input(f"\n⚠️  '{user_id}' foydalanuvchisini tizimdan to'liq o'chirishni xohlaysizmi? [y/N]: ")
    if confirm.lower() != "y":
        print("Bekor qilindi.")
        sys.exit(0)

    delete_user(user_id)
