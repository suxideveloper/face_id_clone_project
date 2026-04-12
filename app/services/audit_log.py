import json
import os
from datetime import datetime
from typing import Any, Optional

from app.core.config import settings

MAX_ENTRIES = 800


def _path():
    return os.path.join(settings.DATA_DIR, "audit_log.json")


def _load() -> list:
    p = _path()
    if not os.path.exists(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"audit_log load error: {e}")
        return []


def _save(entries: list) -> None:
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    with open(_path(), "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False, default=str)


def append_entry(
    action: str,
    actor: str,
    detail: Optional[dict[str, Any]] = None,
) -> None:
    entries = _load()
    entry = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "actor": actor or "noma'lum",
        "action": action,
        "detail": detail or {},
    }
    entries.append(entry)
    if len(entries) > MAX_ENTRIES:
        entries = entries[-MAX_ENTRIES:]
    _save(entries)


def log_attendance_delete(
    actor: str,
    target_date: str,
    worker_id: str,
    worker_name: str,
) -> None:
    append_entry(
        "attendance_delete",
        actor,
        {
            "target_date": target_date,
            "worker_id": worker_id,
            "worker_name": worker_name,
        },
    )


def get_recent(limit: int = 200) -> list:
    entries = _load()
    return list(reversed(entries[-limit:]))
