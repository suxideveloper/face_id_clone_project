from typing import Optional
from app.core.database import SessionLocal
from app.core.models import Holiday


class HolidaysDatabase:
    def get_label(self, date_iso: str) -> Optional[str]:
        with SessionLocal() as db:
            h = db.query(Holiday).filter(Holiday.date == date_iso).first()
            if h:
                return h.label or "Dam olish kuni"
            return None

    def is_holiday(self, date_iso: str) -> bool:
        return self.get_label(date_iso) is not None

    def list_all(self) -> list:
        with SessionLocal() as db:
            rows = db.query(Holiday).order_by(Holiday.date).all()
            return [{"date": h.date, "label": h.label} for h in rows]

    def add(self, date_iso: str, label: str) -> bool:
        label = (label or "").strip() or "Dam olish kuni"
        with SessionLocal() as db:
            existing = db.query(Holiday).filter(Holiday.date == date_iso).first()
            if existing:
                existing.label = label
            else:
                db.add(Holiday(date=date_iso, label=label))
            db.commit()
            return True

    def remove(self, date_iso: str) -> bool:
        with SessionLocal() as db:
            h = db.query(Holiday).filter(Holiday.date == date_iso).first()
            if h:
                db.delete(h)
                db.commit()
                return True
            return False


holidays_db = HolidaysDatabase()
