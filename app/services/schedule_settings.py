import json
import os
from datetime import datetime, date

from app.core.config import settings

DEFAULT_SCHEDULE = {"work_start": "09:00", "work_end": "18:00"}


def _path():
    return os.path.join(settings.DATA_DIR, "schedule_settings.json")


def load_schedule() -> dict:
    p = _path()
    if not os.path.exists(p):
        return dict(DEFAULT_SCHEDULE)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = {**DEFAULT_SCHEDULE, **data}
        return out
    except Exception as e:
        print(f"schedule_settings load error: {e}")
        return dict(DEFAULT_SCHEDULE)


def save_schedule(work_start: str, work_end: str) -> None:
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    data = {"work_start": work_start.strip(), "work_end": work_end.strip()}
    with open(_path(), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _parse_hhmm(s: str) -> tuple:
    parts = (s or "09:00").strip().split(":")
    h = int(parts[0]) if parts else 9
    m = int(parts[1]) if len(parts) > 1 else 0
    return h, m


def work_bounds_for_date(d: date, sched=None) -> tuple:
    """Local work_start and work_end as datetime for calendar day `d`."""
    sched = sched or load_schedule()
    sh, sm = _parse_hhmm(sched.get("work_start", "09:00"))
    eh, em = _parse_hhmm(sched.get("work_end", "18:00"))
    start = datetime(d.year, d.month, d.day, sh, sm, 0)
    end = datetime(d.year, d.month, d.day, eh, em, 0)
    return start, end


def classify_attendance_status(check_in_dt, check_out_dt, d: date) -> dict:
    """
    Returns keys: late (bool), early_leave (bool), label (str), type (str).
    """
    ws, we = work_bounds_for_date(d)
    late = False
    early = False
    if check_in_dt and isinstance(check_in_dt, datetime):
        late = check_in_dt > ws
    if check_out_dt and isinstance(check_out_dt, datetime):
        early = check_out_dt < we

    if check_in_dt and not check_out_dt:
        if late:
            label = "Kechikib kelgan"
            st = "late"
        else:
            label = "Ishda"
            st = "on_time"
        return {"late": late, "early_leave": False, "label": label, "type": st}

    if check_in_dt and check_out_dt:
        if late and early:
            label = "Kechikkan / erta ketgan"
            st = "late_early"
        elif late:
            label = "Kechikkan"
            st = "late"
        elif early:
            label = "Erta ketgan"
            st = "early"
        else:
            label = "To'liq kun"
            st = "complete"
        return {"late": late, "early_leave": early, "label": label, "type": st}

    return {"late": False, "early_leave": False, "label": "Noaniq", "type": "unknown"}
