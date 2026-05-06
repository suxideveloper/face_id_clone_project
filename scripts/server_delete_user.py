#!/usr/bin/env python3
"""
SERVER uchun foydalanuvchini o'chirish skripti.
App import talab qilmaydi — faqat psycopg2 kerak.

Serverda ishlatish:
    cd /path/to/faceid
    python3 scripts/server_delete_user.py Murod
"""

import sys
import os
import pickle
import shutil
import psycopg2

# ── DB ulanish ma'lumotlari (server .env ga mos) ──────────────────────────────
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "faceid_db"
DB_USER = "faceid_user"
DB_PASS = "faceid_pass"
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
ENCODINGS   = os.path.join(DATA_DIR, "encodings.pkl")
IMAGES_DIR  = os.path.join(DATA_DIR, "images")
SNAPS_DIR   = os.path.join(DATA_DIR, "attendance_snapshots")


def delete_user(user_id: str):
    deleted   = []
    not_found = []

    print(f"\n{'='*52}")
    print(f"  O'chirilmoqda: '{user_id}'")
    print(f"{'='*52}")

    # 1. PostgreSQL — users (cascade: face_encodings + attendances ham o'chadi)
    print("\n[1/4] PostgreSQL...")
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT,
            dbname=DB_NAME, user=DB_USER, password=DB_PASS
        )
        cur = conn.cursor()

        # Avval sonlarni olamiz (ma'lumot uchun)
        cur.execute("SELECT COUNT(*) FROM attendances  WHERE username = %s", (user_id,))
        att_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM face_encodings WHERE username = %s", (user_id,))
        enc_count = cur.fetchone()[0]
        cur.execute("SELECT username FROM users WHERE username = %s", (user_id,))
        row = cur.fetchone()

        if row:
            # CASCADE bo'lgani uchun faqat users dan o'chirish yetarli
            cur.execute("DELETE FROM users WHERE username = %s", (user_id,))
            conn.commit()
            deleted.append(f"DB: users + {att_count} davomat + {enc_count} face encoding")
            print(f"  ✅ Foydalanuvchi, {att_count} davomat, {enc_count} encoding o'chirildi")
        else:
            not_found.append("DB: foydalanuvchi topilmadi")
            print(f"  ⚠️  '{user_id}' DB da topilmadi")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"  ❌ DB xatosi: {e}")

    # 2. encodings.pkl — yuz ma'lumotlari
    print("\n[2/4] encodings.pkl...")
    try:
        if os.path.exists(ENCODINGS):
            with open(ENCODINGS, "rb") as f:
                data = pickle.load(f)

            removed = 0
            if isinstance(data, dict):
                if "names" in data and "encodings" in data:
                    # Format: {"names": [...], "encodings": [...]}
                    pairs = [(n, e) for n, e in zip(data["names"], data["encodings"]) if n != user_id]
                    removed = len(data["names"]) - len(pairs)
                    if pairs:
                        names, encs = zip(*pairs)
                        data["names"] = list(names)
                        data["encodings"] = list(encs)
                    else:
                        data["names"] = []
                        data["encodings"] = []
                elif user_id in data:
                    # Format: {user_id: [emb1, emb2, ...]}
                    removed = len(data[user_id])
                    del data[user_id]

            with open(ENCODINGS, "wb") as f:
                pickle.dump(data, f)

            if removed > 0:
                deleted.append(f"encodings.pkl: {removed} ta yuz ma'lumoti")
                print(f"  ✅ {removed} ta encoding o'chirildi")
            else:
                not_found.append("encodings.pkl: encoding topilmadi")
                print(f"  ⚠️  Encoding topilmadi")
        else:
            print(f"  ⚠️  encodings.pkl fayli yo'q")
    except Exception as e:
        print(f"  ❌ Encoding xatosi: {e}")

    # 3. data/images/<user_id>/ — yuz rasmlari
    print("\n[3/4] Yuz rasmlari papkasi...")
    user_img = os.path.join(IMAGES_DIR, user_id)
    try:
        if os.path.exists(user_img):
            count = sum(1 for f in os.listdir(user_img) if os.path.isfile(os.path.join(user_img, f)))
            shutil.rmtree(user_img)
            deleted.append(f"images/{user_id}/: {count} ta rasm")
            print(f"  ✅ {count} ta rasm o'chirildi")
        else:
            not_found.append(f"images/{user_id}/: papka yo'q")
            print(f"  ⚠️  Rasmlar papkasi topilmadi")
    except Exception as e:
        print(f"  ❌ Rasmlar xatosi: {e}")

    # 4. attendance_snapshots — davomat rasmlari
    print("\n[4/4] Davomat snapshot rasmlari...")
    try:
        if os.path.exists(SNAPS_DIR):
            removed_snaps = 0
            for fname in os.listdir(SNAPS_DIR):
                if fname.startswith(f"{user_id}_") or fname.startswith(f"{user_id}."):
                    os.remove(os.path.join(SNAPS_DIR, fname))
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

    # ── Xulosa ────────────────────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"  XULOSA")
    print(f"{'='*52}")
    if deleted:
        print(f"\n  ✅ O'chirildi:")
        for item in deleted:
            print(f"     • {item}")
    if not_found:
        print(f"\n  ⚠️  Topilmadi:")
        for item in not_found:
            print(f"     • {item}")
    print(f"\n  '{user_id}' tizimdan to'liq o'chirildi ✅\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nIshlatish: python3 scripts/server_delete_user.py <user_id>")
        print("Misol:     python3 scripts/server_delete_user.py Murod\n")
        sys.exit(1)

    user_id = sys.argv[1]
    confirm = input(f"\n⚠️  '{user_id}' ni tizimdan o'chirishni tasdiqlaysizmi? [y/N]: ")
    if confirm.strip().lower() != "y":
        print("Bekor qilindi.")
        sys.exit(0)

    delete_user(user_id)
