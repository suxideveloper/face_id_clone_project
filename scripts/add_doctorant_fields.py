"""
Migration: User jadvaliga is_doctorant va staff_rate ustunlarini qo'shish.
Mavjud foydalanuvchilar: is_doctorant=False, staff_rate=1.0
"""
import psycopg2

DB_CONFIG = {
    "dbname": "faceid_db",
    "user": "faceid_user",
    "password": "faceid_pass",
    "host": "localhost",
    "port": 5432,
}

MIGRATIONS = [
    """
    ALTER TABLE users
    ADD COLUMN IF NOT EXISTS is_doctorant BOOLEAN DEFAULT FALSE;
    """,
    """
    ALTER TABLE users
    ADD COLUMN IF NOT EXISTS staff_rate DOUBLE PRECISION DEFAULT 1.0;
    """,
]


def run():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    for sql in MIGRATIONS:
        print(f"Running: {sql.strip()[:60]}...")
        cur.execute(sql)

    # Mavjud yozuvlarda NULL bo'lsa default qo'yish
    cur.execute("UPDATE users SET is_doctorant = FALSE WHERE is_doctorant IS NULL;")
    cur.execute("UPDATE users SET staff_rate = 1.0 WHERE staff_rate IS NULL;")

    cur.close()
    conn.close()
    print("✅ Migration muvaffaqiyatli bajarildi!")


if __name__ == "__main__":
    run()
