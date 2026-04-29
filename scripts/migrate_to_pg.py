import sys
import os
import json
import pickle

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import engine, Base, SessionLocal
from app.core.models import User, FaceEncoding, Attendance, Admin, Department, Holiday, AuditLogEntry


def migrate():
    print("=" * 60)
    print("FaceID: JSON → PostgreSQL migratsiya")
    print("=" * 60)

    print("\n📦 Jadvallar yaratilmoqda...")
    Base.metadata.create_all(bind=engine)
    print("   OK jadvallar tayyor.")

    db = SessionLocal()

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    print(f"   Data papkasi: {data_dir}\n")

    # ── 1. Migrate Admins ──────────────────────────────────────
    admins_file = os.path.join(data_dir, "admins.json")
    if os.path.exists(admins_file):
        print("👤 Adminlar migratsiyasi...")
        with open(admins_file, "r") as f:
            admins_data = json.load(f)
            count = 0
            for username, data in admins_data.items():
                if not db.query(Admin).filter(Admin.username == username).first():
                    # BUG FIX: admins.json da salt va hashed_password alohida
                    # Yangi formatda "salt:hash" qilib saqlash kerak
                    salt = data.get("salt", "")
                    hashed_pw = data.get("hashed_password", "")
                    if salt and hashed_pw:
                        password_hash = f"{salt}:{hashed_pw}"
                    else:
                        password_hash = hashed_pw or ""

                    admin = Admin(
                        username=username,
                        password_hash=password_hash,
                        role=data.get("role", "admin")
                    )
                    db.add(admin)
                    count += 1
        db.commit()
        print(f"   OK {count} ta admin qo'shildi.")
    else:
        print("⚠️  admins.json topilmadi — o'tkazib yuborildi.")

    # ── 2. Migrate Departments ─────────────────────────────────
    depts_file = os.path.join(data_dir, "departments.json")
    if os.path.exists(depts_file):
        print("🏢 Bo'limlar migratsiyasi...")
        with open(depts_file, "r") as f:
            depts_data = json.load(f)
            count = 0
            for dept in depts_data:
                if not db.query(Department).filter(Department.id == dept.get("id")).first():
                    d = Department(
                        id=dept.get("id"),
                        name=dept.get("name"),
                        description=dept.get("description", "")
                    )
                    db.add(d)
                    count += 1
        db.commit()
        print(f"   OK {count} ta bo'lim qo'shildi.")
    else:
        print("⚠️  departments.json topilmadi — o'tkazib yuborildi.")

    # ── 3. Migrate Users ───────────────────────────────────────
    users_file = os.path.join(data_dir, "users.json")
    if os.path.exists(users_file):
        print("👥 Foydalanuvchilar migratsiyasi...")
        with open(users_file, "r") as f:
            users_data = json.load(f)
            count = 0
            for username, data in users_data.items():
                if not db.query(User).filter(User.username == username).first():
                    u = User(
                        username=username,
                        full_name=data.get("full_name", ""),
                        phone=data.get("phone", ""),
                        department=data.get("department", ""),
                        position=data.get("position", ""),
                        registered=data.get("registered", True)
                    )
                    db.add(u)
                    count += 1
        db.commit()
        print(f"   OK {count} ta foydalanuvchi qo'shildi.")
    else:
        print("⚠️  users.json topilmadi — o'tkazib yuborildi.")

    # ── 4. Migrate Face Encodings ──────────────────────────────
    encodings_file = os.path.join(data_dir, "encodings.pkl")
    if os.path.exists(encodings_file):
        print("🧬 Yuz encodinglar migratsiyasi...")
        with open(encodings_file, "rb") as f:
            enc_data = pickle.load(f)
            user_encodings = enc_data.get('user_encodings', {})
            total_enc = 0

            for username, encs in user_encodings.items():
                # Verify user exists
                user = db.query(User).filter(User.username == username).first()
                if not user:
                    print(f"   ⚠️  {username} — users jadvalida topilmadi, o'tkazib yuborildi")
                    continue
                # Only add if no encodings exist for this user yet
                if not db.query(FaceEncoding).filter(FaceEncoding.username == username).first():
                    for enc in encs:
                        e = FaceEncoding(username=username, embedding=enc.tolist())
                        db.add(e)
                        total_enc += 1

            # Handle legacy format: {encodings: [...], names: [...]}
            if not user_encodings and "encodings" in enc_data:
                names = enc_data.get("names", [])
                encs_list = enc_data.get("encodings", [])
                for i, username in enumerate(names):
                    user = db.query(User).filter(User.username == username).first()
                    if not user:
                        continue
                    if not db.query(FaceEncoding).filter(FaceEncoding.username == username).first():
                        e = FaceEncoding(username=username, embedding=encs_list[i].tolist())
                        db.add(e)
                        total_enc += 1

        db.commit()
        print(f"   OK {total_enc} ta encoding qo'shildi.")
    else:
        print("⚠️  encodings.pkl topilmadi — o'tkazib yuborildi.")

    # ── 5. Migrate Attendance ──────────────────────────────────
    # BUG FIX: attendance.json formati {date: {worker_id: record}},
    # oldingi skriptda teskari o'qilgan edi
    attendance_file = os.path.join(data_dir, "attendance.json")
    if os.path.exists(attendance_file):
        print("📋 Davomat migratsiyasi...")
        with open(attendance_file, "r") as f:
            att_data = json.load(f)
            count = 0
            for date_str, workers in att_data.items():
                for worker_id, record in workers.items():
                    # Check if record already exists
                    existing = db.query(Attendance).filter(
                        Attendance.username == worker_id,
                        Attendance.date == date_str
                    ).first()
                    if not existing:
                        a = Attendance(
                            username=worker_id,
                            date=date_str,
                            check_in_time=record.get("check_in_time"),
                            check_out_time=record.get("check_out_time")
                        )
                        db.add(a)
                        count += 1
        db.commit()
        print(f"   OK {count} ta davomat yozuvi qo'shildi.")
    else:
        print("⚠️  attendance.json topilmadi — o'tkazib yuborildi.")

    # ── 6. Migrate Holidays ────────────────────────────────────
    holidays_file = os.path.join(data_dir, "holidays.json")
    if os.path.exists(holidays_file):
        print("🎉 Bayramlar migratsiyasi...")
        with open(holidays_file, "r") as f:
            hol_data = json.load(f)
            count = 0
            # holidays.json formati: {date: label} yoki [{date, label}]
            if isinstance(hol_data, dict):
                for date_str, label in hol_data.items():
                    if not db.query(Holiday).filter(Holiday.date == date_str).first():
                        db.add(Holiday(date=date_str, label=label or "Dam olish kuni"))
                        count += 1
            elif isinstance(hol_data, list):
                for h in hol_data:
                    d = h.get("date", "")
                    if d and not db.query(Holiday).filter(Holiday.date == d).first():
                        db.add(Holiday(date=d, label=h.get("label", "Dam olish kuni")))
                        count += 1
        db.commit()
        print(f"   OK {count} ta bayram qo'shildi.")
    else:
        print("⚠️  holidays.json topilmadi — o'tkazib yuborildi.")

    # ── 7. Migrate Audit Logs ──────────────────────────────────
    # BUG FIX: audit_log.json da maydon nomlari "ts", "actor", "detail"
    # (eskisi "timestamp", "admin_user", "details" deb o'qigan)
    audit_file = os.path.join(data_dir, "audit_log.json")
    if os.path.exists(audit_file):
        print("📝 Audit log migratsiyasi...")
        with open(audit_file, "r") as f:
            audit_data = json.load(f)
            count = 0
            for log in audit_data:
                # Support both old field names (ts/actor/detail) and new (timestamp/admin_user/details)
                ts = log.get("ts") or log.get("timestamp", "")
                actor = log.get("actor") or log.get("admin_user", "")
                action = log.get("action", "")
                detail = log.get("detail") or log.get("details", {})

                a = AuditLogEntry(
                    timestamp=ts,
                    admin_user=actor,
                    action=action,
                    details=json.dumps(detail, ensure_ascii=False, default=str) if isinstance(detail, dict) else str(detail)
                )
                db.add(a)
                count += 1
        db.commit()
        print(f"   OK {count} ta audit log qo'shildi.")
    else:
        print("⚠️  audit_log.json topilmadi — o'tkazib yuborildi.")

    db.close()

    print("\n" + "=" * 60)
    print("✅ MIGRATSIYA MUVAFFAQIYATLI YAKUNLANDI!")
    print("=" * 60)
    print("\nKeyingi qadam: python scripts/test_db.py")


if __name__ == "__main__":
    migrate()
