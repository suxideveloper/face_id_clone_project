"""
export_to_json.py
=================
PostgreSQL dagi barcha ma'lumotlarni JSON fayllariga eksport qiladi.
Ishlatish:  python scripts/export_to_json.py
"""

import sys
import os
import json
import pickle
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.core.models import User, FaceEncoding, Attendance, Admin, Department, Holiday, AuditLogEntry

# ── Backup papkasini tayyorlash ────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_DIR = os.path.join(BASE_DIR, "backups", TIMESTAMP)

os.makedirs(BACKUP_DIR, exist_ok=True)

def save_json(data, filename):
    path = os.path.join(BACKUP_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"   ✅ {filename} saqlandi → {path}")
    return path

def export():
    print("=" * 60)
    print("📦 FaceID: PostgreSQL → JSON eksport")
    print(f"📁 Backup papka: {BACKUP_DIR}")
    print("=" * 60)

    db = SessionLocal()

    try:
        # ── 1. Users ──────────────────────────────────────────────
        print("\n👥 Foydalanuvchilar eksport...")
        users = db.query(User).all()
        users_data = {}
        for u in users:
            users_data[u.username] = {
                "full_name":    u.full_name,
                "phone":        u.phone,
                "department":   u.department,
                "position":     u.position,
                "registered":   u.registered,
                "is_doctorant": u.is_doctorant,
                "staff_rate":   u.staff_rate,
            }
        save_json(users_data, "users.json")
        print(f"   {len(users)} ta foydalanuvchi")

        # ── 2. Admins ─────────────────────────────────────────────
        print("\n👤 Adminlar eksport...")
        admins = db.query(Admin).all()
        admins_data = {}
        for a in admins:
            # password_hash "salt:hash" formatida — ajratib saqlaymiz
            parts = a.password_hash.split(":", 1) if a.password_hash else ["", ""]
            admins_data[a.username] = {
                "salt":            parts[0] if len(parts) == 2 else "",
                "hashed_password": parts[1] if len(parts) == 2 else parts[0],
                "role":            a.role,
            }
        save_json(admins_data, "admins.json")
        print(f"   {len(admins)} ta admin")

        # ── 3. Departments ────────────────────────────────────────
        print("\n🏢 Bo'limlar eksport...")
        depts = db.query(Department).all()
        depts_data = [
            {"id": d.id, "name": d.name, "description": d.description}
            for d in depts
        ]
        save_json(depts_data, "departments.json")
        print(f"   {len(depts)} ta bo'lim")

        # ── 4. Attendance ─────────────────────────────────────────
        print("\n📋 Davomat eksport...")
        records = db.query(Attendance).all()
        att_data = {}
        for r in records:
            if r.date not in att_data:
                att_data[r.date] = {}
            att_data[r.date][r.username] = {
                "check_in_time":  r.check_in_time,
                "check_out_time": r.check_out_time,
            }
        save_json(att_data, "attendance.json")
        print(f"   {len(records)} ta davomat yozuvi")

        # ── 5. Holidays ───────────────────────────────────────────
        print("\n🎉 Bayramlar eksport...")
        holidays = db.query(Holiday).all()
        hol_data = {h.date: h.label for h in holidays}
        save_json(hol_data, "holidays.json")
        print(f"   {len(holidays)} ta bayram")

        # ── 6. Audit Logs ─────────────────────────────────────────
        print("\n📝 Audit loglar eksport...")
        logs = db.query(AuditLogEntry).order_by(AuditLogEntry.id).all()
        log_data = [
            {
                "ts":     l.timestamp,
                "actor":  l.admin_user,
                "action": l.action,
                "detail": l.details,
            }
            for l in logs
        ]
        save_json(log_data, "audit_log.json")
        print(f"   {len(logs)} ta log yozuvi")

        # ── 7. Face Encodings (PKL) ───────────────────────────────
        print("\n🧬 Yuz encodinglar eksport (encodings.pkl)...")
        enc_rows = db.query(FaceEncoding).all()
        user_encodings = {}
        for row in enc_rows:
            if row.username not in user_encodings:
                user_encodings[row.username] = []
            user_encodings[row.username].append(np.array(row.embedding))

        pkl_path = os.path.join(BACKUP_DIR, "encodings.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"user_encodings": user_encodings}, f)
        print(f"   ✅ encodings.pkl saqlandi → {pkl_path}")
        total_enc = sum(len(v) for v in user_encodings.values())
        print(f"   {len(user_encodings)} ta foydalanuvchi, {total_enc} ta embedding")

        # ── 8. data/ papkaga ham nusxa ko'chirish ─────────────────
        import shutil
        print("\n🔄 data/ papkaga ham nusxa ko'chirilmoqda...")
        for fname in ["users.json", "admins.json", "departments.json",
                      "attendance.json", "holidays.json", "audit_log.json"]:
            src = os.path.join(BACKUP_DIR, fname)
            dst = os.path.join(DATA_DIR, fname)
            shutil.copy2(src, dst)
            print(f"   ✅ data/{fname} yangilandi")

        shutil.copy2(pkl_path, os.path.join(DATA_DIR, "encodings.pkl"))
        print("   ✅ data/encodings.pkl yangilandi")

    finally:
        db.close()

    print("\n" + "=" * 60)
    print("✅ EKSPORT MUVAFFAQIYATLI YAKUNLANDI!")
    print(f"📁 Backup: {BACKUP_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    export()
