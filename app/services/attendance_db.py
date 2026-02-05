import json
import os
from datetime import datetime, date
from app.core.config import settings

class AttendanceDatabase:
    """
    Rolling attendance database that tracks check-in and check-out times.
    Stores the first recognition as check-in and updates subsequent ones as check-out.
    Automatically cleans up old check-out snapshots to save storage.
    """
    
    def __init__(self):
        self.db_path = os.path.join(settings.DATA_DIR, "attendance.json")
        self.snapshots_dir = os.path.join(settings.DATA_DIR, "attendance_snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)
        self.records = {}  # {date_str: {worker_id: record}}
        self.load()
    
    def load(self):
        """Load attendance records from disk."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    self.records = json.load(f)
            except Exception as e:
                print(f"Error loading attendance database: {e}")
                self.records = {}
        else:
            self.records = {}
    
    def save(self):
        """Save attendance records to disk."""
        with open(self.db_path, 'w') as f:
            json.dump(self.records, f, indent=2, default=str)
    
    def _get_today_key(self) -> str:
        """Get today's date as string key."""
        return date.today().isoformat()
    
    def get_today_record(self, worker_id: str) -> dict:
        """Get today's attendance record for a worker."""
        today = self._get_today_key()
        if today in self.records and worker_id in self.records[today]:
            return self.records[today][worker_id]
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
        
        # Initialize today's records if not exists
        if today not in self.records:
            self.records[today] = {}
        
        existing_record = self.records[today].get(worker_id)
        
        if existing_record is None:
            # Case A: New entry - Check In
            snapshot_path = self._save_snapshot(worker_id, frame, "checkin")
            
            self.records[today][worker_id] = {
                "worker_id": worker_id,
                "date": today,
                "check_in_time": now.isoformat(),
                "check_out_time": None,
                "check_in_snapshot": snapshot_path,
                "check_out_snapshot": None
            }
            self.save()
            
            return {
                "event_type": "check_in",
                "record": self.records[today][worker_id]
            }
        else:
            # Case B: Update entry - Check Out
            # Cleanup old check_out snapshot before saving new one
            old_snapshot = existing_record.get("check_out_snapshot")
            self._cleanup_old_snapshot(old_snapshot)
            
            # Save new snapshot
            new_snapshot_path = self._save_snapshot(worker_id, frame, "checkout")
            
            # Update record
            existing_record["check_out_time"] = now.isoformat()
            existing_record["check_out_snapshot"] = new_snapshot_path
            self.save()
            
            return {
                "event_type": "check_out",
                "record": existing_record
            }
    
    def get_all_today(self) -> dict:
        """Get all attendance records for today."""
        today = self._get_today_key()
        return self.records.get(today, {})
    
    def get_records_by_date(self, date_str: str) -> dict:
        """Get all attendance records for a specific date."""
        return self.records.get(date_str, {})
    
    def get_worker_history(self, worker_id: str, limit: int = 30) -> list:
        """Get attendance history for a worker (last N days)."""
        history = []
        for date_str in sorted(self.records.keys(), reverse=True)[:limit]:
            if worker_id in self.records[date_str]:
                history.append(self.records[date_str][worker_id])
        return history
    
    def get_all_records(self, start_date=None, end_date=None) -> list:
        """Get flattened list of all records, optionally filtered by date range."""
        all_records = []
        
        # Sort dates descending (newest first)
        for date_str in sorted(self.records.keys(), reverse=True):
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue
                
            day_records = self.records[date_str]
            for worker_id, record in day_records.items():
                all_records.append(record)
                
        return all_records



    def delete_record(self, date_str: str, worker_id: str) -> bool:
        """Delete an attendance record and associated snapshots."""
        if date_str in self.records and worker_id in self.records[date_str]:
            record = self.records[date_str][worker_id]
            
            # Cleanup snapshots
            self._cleanup_old_snapshot(record.get("check_in_snapshot"))
            self._cleanup_old_snapshot(record.get("check_out_snapshot"))
            
            # Remove record from memory
            del self.records[date_str][worker_id]
            
            # If date empty, remove date
            if not self.records[date_str]:
                del self.records[date_str]
                
            self.save()
            return True
        return False

# Singleton instance
attendance_db = AttendanceDatabase()
