import os
from datetime import datetime, date
from app.core.config import settings
from app.core.database import SessionLocal
from app.core.models import Attendance


class AttendanceDatabase:
    """
    Rolling attendance database that tracks check-in and check-out times.
    Stores the first recognition as check-in and updates subsequent ones as check-out.
    Automatically cleans up old check-out snapshots to save storage.
    """

    def __init__(self):
        self.snapshots_dir = os.path.join(settings.DATA_DIR, "attendance_snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def _get_today_key(self) -> str:
        """Get today's date as string key."""
        return date.today().isoformat()

    def get_today_record(self, worker_id: str) -> dict:
        """Get today's attendance record for a worker."""
        today = self._get_today_key()
        with SessionLocal() as db:
            rec = db.query(Attendance).filter(
                Attendance.username == worker_id,
                Attendance.date == today
            ).first()
            if rec:
                return self._row_to_dict(rec)
            return None

    def _row_to_dict(self, rec: Attendance) -> dict:
        """Convert a DB row to the legacy dict format."""
        return {
            "worker_id": rec.username,
            "date": rec.date,
            "check_in_time": rec.check_in_time,
            "check_out_time": rec.check_out_time,
            # Snapshots are still stored on disk; we derive paths from convention
            "check_in_snapshot": self._snapshot_path(rec.username, rec.date, "checkin"),
            "check_out_snapshot": self._snapshot_path(rec.username, rec.date, "checkout"),
        }

    def _snapshot_path(self, worker_id, date_str, stype):
        """Return the first matching snapshot file on disk (or None)."""
        prefix = f"{worker_id}_{date_str}_{stype}_"
        try:
            for fn in os.listdir(self.snapshots_dir):
                if fn.startswith(prefix):
                    return os.path.join(self.snapshots_dir, fn)
        except Exception:
            pass
        return None

    def _cleanup_old_snapshot(self, filepath: str):
        """Delete old snapshot file to save storage."""
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Error cleaning up snapshot: {e}")

    def _save_snapshot(self, worker_id: str, frame, snapshot_type: str) -> str:
        """Save a snapshot image and return the file path."""
        import cv2

        today = self._get_today_key()
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"{worker_id}_{today}_{snapshot_type}_{timestamp}.jpg"
        filepath = os.path.join(self.snapshots_dir, filename)

        try:
            cv2.imwrite(filepath, frame)
            return filepath
        except Exception as e:
            print(f"Error saving snapshot: {e}")
            return None

    def record_attendance(self, worker_id: str, frame) -> dict:
        """
        Record attendance for a worker.

        Returns:
            dict with 'event_type' ('check_in' or 'check_out') and record data
        """
        today = self._get_today_key()
        now = datetime.now()

        with SessionLocal() as db:
            existing = db.query(Attendance).filter(
                Attendance.username == worker_id,
                Attendance.date == today
            ).first()

            if existing is None:
                # Case A: New entry – Check In
                snapshot_path = self._save_snapshot(worker_id, frame, "checkin")

                rec = Attendance(
                    username=worker_id,
                    date=today,
                    check_in_time=now.isoformat(),
                    check_out_time=None,
                )
                db.add(rec)
                db.commit()

                return {
                    "event_type": "check_in",
                    "record": self._row_to_dict(rec),
                }
            else:
                # Case B: Update entry – Check Out
                old_snapshot = self._snapshot_path(worker_id, today, "checkout")
                self._cleanup_old_snapshot(old_snapshot)

                self._save_snapshot(worker_id, frame, "checkout")

                existing.check_out_time = now.isoformat()
                db.commit()

                return {
                    "event_type": "check_out",
                    "record": self._row_to_dict(existing),
                }

    def get_all_today(self) -> dict:
        """Get all attendance records for today."""
        today = self._get_today_key()
        with SessionLocal() as db:
            rows = db.query(Attendance).filter(Attendance.date == today).all()
            return {r.username: self._row_to_dict(r) for r in rows}

    def get_records_by_date(self, date_str: str) -> dict:
        """Get all attendance records for a specific date."""
        with SessionLocal() as db:
            rows = db.query(Attendance).filter(Attendance.date == date_str).all()
            return {r.username: self._row_to_dict(r) for r in rows}

    def get_worker_history(self, worker_id: str, limit: int = 30) -> list:
        """Get attendance history for a worker (last N days)."""
        with SessionLocal() as db:
            rows = (
                db.query(Attendance)
                .filter(Attendance.username == worker_id)
                .order_by(Attendance.date.desc(), Attendance.check_in_time.desc())
                .limit(limit)
                .all()
            )
            return [self._row_to_dict(r) for r in rows]

    def get_all_records(self, start_date=None, end_date=None) -> list:
        """Get flattened list of all records, optionally filtered by date range."""
        with SessionLocal() as db:
            q = db.query(Attendance)
            if start_date:
                q = q.filter(Attendance.date >= start_date)
            if end_date:
                q = q.filter(Attendance.date <= end_date)
            rows = q.order_by(Attendance.date.desc(), Attendance.check_in_time.desc()).all()
            return [self._row_to_dict(r) for r in rows]

    def delete_record(self, date_str: str, worker_id: str) -> bool:
        """Delete an attendance record and associated snapshots."""
        with SessionLocal() as db:
            rec = db.query(Attendance).filter(
                Attendance.username == worker_id,
                Attendance.date == date_str
            ).first()
            if rec:
                self._cleanup_old_snapshot(self._snapshot_path(worker_id, date_str, "checkin"))
                self._cleanup_old_snapshot(self._snapshot_path(worker_id, date_str, "checkout"))
                db.delete(rec)
                db.commit()
                return True
            return False


# Singleton instance
attendance_db = AttendanceDatabase()
