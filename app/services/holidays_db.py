import json
import os
from typing import Optional

from app.core.config import settings


class HolidaysDatabase:
    def __init__(self):
        self.db_path = os.path.join(settings.DATA_DIR, "holidays.json")
        self.holidays: list = []
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.holidays = json.load(f)
                if not isinstance(self.holidays, list):
                    self.holidays = []
            except Exception as e:
                print(f"holidays load error: {e}")
                self.holidays = []
        else:
            self.holidays = []

    def save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.holidays, f, indent=2, ensure_ascii=False)

    def get_label(self, date_iso: str) -> Optional[str]:
        for h in self.holidays:
            if h.get("date") == date_iso:
                return h.get("label") or "Dam olish kuni"
        return None

    def is_holiday(self, date_iso: str) -> bool:
        return self.get_label(date_iso) is not None

    def list_all(self) -> list:
        return sorted(self.holidays, key=lambda x: x.get("date", ""))

    def add(self, date_iso: str, label: str) -> bool:
        label = (label or "").strip() or "Dam olish kuni"
        for h in self.holidays:
            if h.get("date") == date_iso:
                h["label"] = label
                self.save()
                return True
        self.holidays.append({"date": date_iso, "label": label})
        self.save()
        return True

    def remove(self, date_iso: str) -> bool:
        before = len(self.holidays)
        self.holidays = [h for h in self.holidays if h.get("date") != date_iso]
        if len(self.holidays) < before:
            self.save()
            return True
        return False


holidays_db = HolidaysDatabase()
