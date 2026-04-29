import sys
sys.path.insert(0, '.')

print('=' * 60)
print('1. PostgreSQL ULANISH TESTI')
print('=' * 60)
from app.core.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text('SELECT version()')).fetchone()
    print(f'   OK PostgreSQL ishlayapti: {result[0][:50]}...')

print()
print('=' * 60)
print('2. pgvector KENGAYTMASI TESTI')
print('=' * 60)
with engine.connect() as conn:
    result = conn.execute(text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")).fetchone()
    if result:
        print(f'   OK pgvector kengaytmasi faol: versiya {result[1]}')
    else:
        print('   FAIL pgvector kengaytmasi TOPILMADI!')

print()
print('=' * 60)
print('3. JADVALLAR TESTI')
print('=' * 60)
with engine.connect() as conn:
    tables = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")).fetchall()
    for t in tables:
        count = conn.execute(text(f'SELECT count(*) FROM {t[0]}')).fetchone()[0]
        print(f'   {t[0]}: {count} ta yozuv')

print()
print('=' * 60)
print('4. FACE ENCODINGS va VECTOR USTUN TESTI')
print('=' * 60)
with engine.connect() as conn:
    col_info = conn.execute(text("SELECT column_name, data_type, udt_name FROM information_schema.columns WHERE table_name = 'face_encodings' AND column_name = 'embedding'")).fetchone()
    if col_info:
        print(f'   OK embedding ustuni mavjud: type={col_info[2]}')
    else:
        print('   FAIL embedding ustuni TOPILMADI!')
    
    idx = conn.execute(text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'face_encodings' AND indexdef LIKE '%hnsw%'")).fetchone()
    if idx:
        print(f'   OK HNSW indeks mavjud: {idx[0]}')
    else:
        print('   FAIL HNSW indeks TOPILMADI!')

    rows = conn.execute(text('SELECT username, embedding FROM face_encodings LIMIT 2')).fetchall()
    for r in rows:
        vec = str(r[1])
        print(f'   {r[0]}: vektor = [{vec[:80]}...]')

print()
print('=' * 60)
print('5. pgvector L2 DISTANCE QIDIRUV TESTI (eng muhimi!)')
print('=' * 60)
with engine.connect() as conn:
    sample = conn.execute(text('SELECT username, embedding FROM face_encodings LIMIT 1')).fetchone()
    if sample:
        vec_str = sample[1]
        result = conn.execute(
            text("SELECT username, embedding <-> :vec AS distance FROM face_encodings ORDER BY embedding <-> :vec LIMIT 3"),
            {'vec': vec_str}
        ).fetchall()
        print(f'   Test: "{sample[0]}" vektoriga eng yaqin natijalar:')
        for r in result:
            status = 'MATCH' if r[1] < 0.45 else 'TOO FAR'
            print(f'      -> {r[0]}: distance={r[1]:.6f} [{status}]')
        print()
        print(f'   OK pgvector <-> (L2 distance) operatori ISHLAYAPTI!')
    else:
        print('   WARNING: Bazada encoding topilmadi')

print()
print('=' * 60)
print('6. SERVISLAR TESTI')
print('=' * 60)
from app.services.user_db import user_db
from app.services.attendance_db import attendance_db
from app.services.departments_db import departments_db
from app.services.holidays_db import holidays_db
from app.services.recognizer import recognizer
from app.services import admin_db

users = user_db.get_all_users()
print(f'   OK user_db: {len(users)} foydalanuvchi')
depts = departments_db.get_all()
print(f'   OK departments_db: {len(depts)} bolim')
names = recognizer.get_all_user_names()
print(f'   OK recognizer: {names}')
verified = admin_db.verify_admin('admin', 'admin')
if verified:
    print(f'   OK admin_db.verify_admin("admin","admin"): Login muvaffaqiyatli')
else:
    print(f'   FAIL admin_db.verify_admin("admin","admin"): Login ishlamadi')

print()
print('=' * 60)
print('XULOSA: Barcha testlar yakunlandi')
print('=' * 60)
