import json
from datetime import datetime
from typing import Any, Optional

from app.core.database import SessionLocal
from app.core.models import AuditLogEntry

MAX_ENTRIES = 800


def append_entry(
    action: str,
    actor: str,
    detail: Optional[dict[str, Any]] = None,
) -> None:
    with SessionLocal() as db:
        entry = AuditLogEntry(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            admin_user=actor or "noma'lum",
            action=action,
            details=json.dumps(detail or {}, ensure_ascii=False, default=str),
        )
        db.add(entry)
        db.commit()

        # Trim to MAX_ENTRIES (keep the newest rows)
        total = db.query(AuditLogEntry).count()
        if total > MAX_ENTRIES:
            excess = total - MAX_ENTRIES
            oldest_ids = (
                db.query(AuditLogEntry.id)
                .order_by(AuditLogEntry.id.asc())
                .limit(excess)
                .subquery()
            )
            db.query(AuditLogEntry).filter(AuditLogEntry.id.in_(oldest_ids)).delete(synchronize_session=False)
            db.commit()


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
    with SessionLocal() as db:
        rows = (
            db.query(AuditLogEntry)
            .order_by(AuditLogEntry.id.desc())
            .limit(limit)
            .all()
        )
        result = []
        for r in rows:
            detail = r.details
            if isinstance(detail, str):
                try:
                    detail = json.loads(detail)
                except Exception:
                    pass
            result.append({
                "ts": r.timestamp,
                "actor": r.admin_user,
                "action": r.action,
                "detail": detail,
            })
        return result
