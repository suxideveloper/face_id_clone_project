from fastapi import APIRouter, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from app.services.camera import camera_service
from app.services.detector import detector
from app.services.recognizer import recognizer
import cv2
import numpy as np
import time
from datetime import datetime, timedelta
import asyncio
import os
from app.core.config import settings

from app.services.tracker import Tracker
from app.services.user_db import user_db
from app.services.attendance_db import attendance_db
from app.services.websocket_manager import ws_manager

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Attendance event queue for async processing
attendance_queue = asyncio.Queue()

# Debounce tracking: {worker_id: last_logged_time}
attendance_debounce = {}
visual_debounce = {}
DEBOUNCE_SECONDS = 30  # Only log to DB once per 30 seconds
VISUAL_DEBOUNCE_SECONDS = 3  # Show visual feedback every 3 seconds

# Initialize Tracker
face_tracker = Tracker()

# State to track current registration progress for UI feedback
# stage: -1 (none), 0 (frontal), 1 (left), 2 (right), 3 (up), 4 (down)
registration_state = {"name": "", "stage": -1, "hold_progress": 0.0}

# Generator for video stream
def generate_frames(mode="verification"):
    while True:
        frame = camera_service.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
            
        h, w = frame.shape[:2]
        
        # Detect faces
        detections = detector.detect(frame)
        
        if mode == "registration":
            # [REGISTRATION LOGIC REMAINS LARGELY SAME BUT UPDATED TO USE DETECTIONS]
            # ... (keeping current high-tech registration UI as is) ...
            COLOR_CYAN = (255, 255, 0)
            COLOR_MAGENTA = (120, 10, 255)
            COLOR_WHITE = (255, 255, 255)
            
            bracket_w, bracket_h = int(w * 0.45), int(h * 0.7)
            bx1, by1 = (w - bracket_w) // 2, (h - bracket_h) // 2
            bx2, by2 = bx1 + bracket_w, by1 + bracket_h
            
            face_in_frame = False
            best_face = None
            for (x1, y1, x2, y2), conf in detections:
                fx, fy = (x1 + x2) // 2, (y1 + y2) // 2
                if bx1 < fx < bx2 and by1 < fy < by2:
                    face_in_frame = True
                    best_face = (x1, y1, x2, y2)
                    break
            
            l_size = 40; t = 3
            cv2.line(frame, (bx1, by1), (bx1 + l_size, by1), COLOR_MAGENTA, t)
            cv2.line(frame, (bx1, by1), (bx1, by1 + l_size), COLOR_MAGENTA, t)
            cv2.line(frame, (bx2, by1), (bx2 - l_size, by1), COLOR_MAGENTA, t)
            cv2.line(frame, (bx2, by1), (bx2, by1 + l_size), COLOR_MAGENTA, t)
            cv2.line(frame, (bx1, by2), (bx1 + l_size, by2), COLOR_MAGENTA, t)
            cv2.line(frame, (bx1, by2), (bx1, by2 - l_size), COLOR_MAGENTA, t)
            cv2.line(frame, (bx2, by2), (bx2 - l_size, by2), COLOR_MAGENTA, t)
            cv2.line(frame, (bx2, by2), (bx2, by2 - l_size), COLOR_MAGENTA, t)

            # Pose instruction data with icons
            stages = [
                ("LOOK STRAIGHT", "[ @ ]"),      # Front
                ("TURN HEAD LEFT", "[ <-- ]"),   # Left
                ("TURN HEAD RIGHT", "[ --> ]"),  # Right
                ("TILT CHIN UP", "[ ^ ]"),       # Up
                ("TILT CHIN DOWN", "[ v ]")      # Down
            ]
            curr_stage = registration_state["stage"]
            
            # Draw dark strip at bottom for text readability
            cv2.rectangle(frame, (0, h - 100), (w, h), (30, 30, 30), -1)
            
            if curr_stage >= 0 and curr_stage < 5:
                # During active registration - show current pose instruction
                stage_text, stage_icon = stages[curr_stage]
                
                # Progress dots (top of dark strip)
                dot_y = h - 85
                dot_start_x = (w - 150) // 2
                for i in range(5):
                    dot_x = dot_start_x + i * 35
                    if i < curr_stage:
                        cv2.circle(frame, (dot_x, dot_y), 8, (0, 255, 0), -1)  # Complete - green
                    elif i == curr_stage:
                        cv2.circle(frame, (dot_x, dot_y), 10, (0, 200, 255), -1)  # Active - orange
                    else:
                        cv2.circle(frame, (dot_x, dot_y), 6, (80, 80, 80), -1)  # Pending - gray
                
                # Get hold progress from state
                hold_progress = registration_state.get("hold_progress", 0.0)
                
                # Draw hold progress bar if face is in frame
                if face_in_frame and hold_progress > 0:
                    bar_width = int(w * 0.4)
                    bar_height = 15
                    bar_x = (w - bar_width) // 2
                    bar_y = by2 + 20
                    # Background
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                    # Progress fill
                    fill_width = int(bar_width * min(hold_progress, 1.0))
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
                    # Border
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
                
                # Pose instruction text (large)
                inst_text = f"STEP {curr_stage + 1}/5: {stage_text}"
                if face_in_frame:
                    color = (0, 255, 0)  # Green - correct position
                    status_text = f"HOLD STILL... {int(hold_progress * 100)}%" if hold_progress > 0 else "DETECTED! HOLD STILL..."
                else:
                    color = (0, 0, 255)  # Red - need to position face
                    status_text = "MOVE INTO FRAME!"
                
                # Main instruction (centered)
                tw = cv2.getTextSize(inst_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0][0]
                cv2.putText(frame, inst_text, ((w - tw) // 2, h - 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv2.LINE_AA)
                
                # Status text below
                sw = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
                cv2.putText(frame, status_text, ((w - sw) // 2, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                
                # Icon in center of bracket area
                iw = cv2.getTextSize(stage_icon, cv2.FONT_HERSHEY_DUPLEX, 2.0, 3)[0][0]
                cv2.putText(frame, stage_icon, ((w - iw) // 2, by1 + 60), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 3, cv2.LINE_AA)
                
            elif curr_stage >= 5:
                # Complete
                cv2.putText(frame, "ALL POSES CAPTURED!", ((w - 400) // 2, h - 50), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                # Before registration starts - show ready status
                if face_in_frame:
                    inst_text = "READY FOR REGISTRATION"
                    color = (0, 255, 0)  # Green
                else:
                    inst_text = "POSITION YOUR FACE IN THE FRAME"
                    color = (0, 165, 255)  # Orange

                tw = cv2.getTextSize(inst_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0][0]
                cv2.putText(frame, inst_text, ((w - tw) // 2, h - 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)

            # Face corner indicators
            if face_in_frame and best_face:
                x1, y1, x2, y2 = best_face
                pulse = int(4 + 2 * np.sin(time.time() * 4))
                face_color = (0, 255, 0) if curr_stage >= 0 else COLOR_CYAN
                cv2.circle(frame, (x1, y1), pulse, face_color, -1)
                cv2.circle(frame, (x2, y1), pulse, face_color, -1)
                cv2.circle(frame, (x1, y2), pulse, face_color, -1)
                cv2.circle(frame, (x2, y2), pulse, face_color, -1)

        else:
            # TRACKING & RECOGNITION MODE
            try:
                # Convert detections to bboxes for the tracker
                bbox_list = []
                for det in detections:
                    # det is ((x1, y1, x2, y2), conf)
                    bbox, conf = det
                    bbox_list.append(bbox)
                
                tracked_faces = face_tracker.update(bbox_list)
                
                for face in tracked_faces:
                    tid = face["id"]
                    x1, y1, x2, y2 = face["bbox"]
                    name = face["name"]
                    needs_reverify = face.get("needs_reverify", False)
                    
                    # Recognition: if name not cached OR needs re-verification
                    if name is None or needs_reverify:
                        try:
                            name = recognizer.verify(frame, (x1, y1, x2, y2))
                            face_tracker.set_name(tid, name)
                        except Exception:
                            name = "Unknown"
                            face_tracker.set_name(tid, name)
                    
                    # Attendance logging & Visual Feedback
                    # Allow "Unknown" to pass through for visual feedback (red alert)
                    if name:
                        current_time = time.time()
                        last_db = attendance_debounce.get(name, 0)
                        last_visual = visual_debounce.get(name, 0)
                        
                        # Logic for Known Users
                        if name != "Unknown":
                            should_log_db = (current_time - last_db >= DEBOUNCE_SECONDS)
                            should_show_visual = (current_time - last_visual >= VISUAL_DEBOUNCE_SECONDS)
                            
                            if should_show_visual:
                                visual_debounce[name] = current_time
                                if should_log_db:
                                    attendance_debounce[name] = current_time
                                
                                try:
                                    attendance_queue.put_nowait({
                                        "worker_id": name,
                                        "frame": frame.copy(),
                                        "log_db": should_log_db
                                    })
                                except asyncio.QueueFull:
                                    pass
                        
                        # Logic for Unknown Users
                        else:
                            # For unknown, we don't log to DB, but we want to trigger visual feedback
                            # Use visual debounce to avoid flooding WebSocket
                            should_show_visual = (current_time - last_visual >= VISUAL_DEBOUNCE_SECONDS)
                            
                            if should_show_visual:
                                visual_debounce[name] = current_time
                                try:
                                    attendance_queue.put_nowait({
                                        "worker_id": "Unknown",
                                        "frame": frame.copy(), # Frame might be needed if we want to save unknown faces later
                                        "log_db": False
                                    })
                                except asyncio.QueueFull:
                                    pass
                    
                    # Draw Results only if recognized
                    # Draw Results (Verified = Green, Unknown = Red)
                    if name:
                        if name == "Unknown":
                            color = (0, 0, 255) # Red
                            label = "Unknown"
                            text_color = (255, 255, 255) # White text
                        else:
                            color = (0, 255, 0) # Green
                            label = f"ID:{tid} | {name}"
                            text_color = (0, 0, 0) # Black text

                        # High-tech corner bounding box
                        t = 2; l = 20
                        cv2.line(frame, (x1, y1), (x1+l, y1), color, t)
                        cv2.line(frame, (x1, y1), (x1, y1+l), color, t)
                        cv2.line(frame, (x2, y1), (x2-l, y1), color, t)
                        cv2.line(frame, (x2, y1), (x2, y1+l), color, t)
                        cv2.line(frame, (x1, y2), (x1+l, y2), color, t)
                        cv2.line(frame, (x1, y2), (x1, y2-l), color, t)
                        cv2.line(frame, (x2, y2), (x2-l, y2), color, t)
                        cv2.line(frame, (x2, y2), (x2, y2-l), color, t)
                        
                        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)[0]
                        cv2.rectangle(frame, (x1, y1-30), (x1 + t_size[0] + 10, y1), color, -1)
                        cv2.putText(frame, label, (x1+5, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, text_color, 1, cv2.LINE_AA)
            except Exception as e:
                # Log error but don't crash the stream
                print(f"Tracking error: {e}")
            
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

from app.services import admin_db
import secrets

# Admin Session Management
active_sessions = {}  # token: username

def get_current_admin(request: Request):
    session_id = request.cookies.get("admin_session")
    if not session_id or session_id not in active_sessions:
        return None
    return active_sessions[session_id]

@router.get("/")
async def index(request: Request):
    active_users = len(recognizer.user_encodings)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "active_users_count": active_users
    })

@router.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/login")
async def login_action(request: Request, username: str = Form(...), password: str = Form(...)):
    if admin_db.verify_admin(username, password):
        # Create session
        token = secrets.token_hex(16)
        active_sessions[token] = username
        
        response = RedirectResponse(url="/admin", status_code=303)
        response.set_cookie(key="admin_session", value=token, httponly=True)
        return response
    else:
        return templates.TemplateResponse("login.html", {
            "request": request, 
            "error": "Invalid username or password"
        })

@router.get("/logout")
async def logout(request: Request):
    session_id = request.cookies.get("admin_session")
    if session_id in active_sessions:
        del active_sessions[session_id]
    
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("admin_session")
    return response

@router.get("/admin")
async def admin_dashboard(request: Request):
    user = get_current_admin(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    
    # --- Statistics Calculation ---
    
    # 1. Total Employees
    # specific to user_db based on how list_users does it
    total_employees = len(user_db.get_all_users())
    
    # 2. Attendance Stats
    now = datetime.now()
    today_date = now.date()
    
    # Helper for counting unique check-ins in a date range
    def count_checkins_since(days_ago: int):
        count = 0
        start_date = (today_date - timedelta(days=days_ago)).isoformat()
        
        # Get all records since start_date
        records = attendance_db.get_all_records(start_date=start_date)
        
        # Count records (assuming one record per person per day is what we want to count)
        count = len(records)
        return count

    weekly_attendance = count_checkins_since(7)
    monthly_attendance = count_checkins_since(30)
    
    # Calculate Attendance Percentage (vs Total Employees)
    attendance_rate = 0
    present_count = 0
    late_count = 0
    on_time_count = 0
    absent_count = 0
    
    if total_employees > 0:
        # Get unique workers today
        today_records = attendance_db.get_all_today()
        present_count = len(today_records)
        attendance_rate = int((present_count / total_employees) * 100)
        
        # Calculate On Time vs Late (Assume 9:00 AM start time)
        # We need to parse check_in_time for each record
        for name, record in today_records.items():
            check_in_str = record.get("check_in_time")
            if check_in_str:
                try:
                    check_in_dt = datetime.fromisoformat(check_in_str)
                    # Create 9 AM threshold for that day
                    nine_am = check_in_dt.replace(hour=9, minute=0, second=0, microsecond=0)
                    
                    if check_in_dt > nine_am:
                        late_count += 1
                    else:
                        on_time_count += 1
                except ValueError:
                    pass
        
        absent_count = total_employees - present_count

    # 3. Growth (Today vs Yesterday)
    yesterday_date = (today_date - timedelta(days=1)).isoformat()
    yesterday_records = attendance_db.get_records_by_date(yesterday_date)
    yesterday_count = len(yesterday_records)
    
    attendance_growth = 0
    if yesterday_count > 0:
        attendance_growth = int(((present_count - yesterday_count) / yesterday_count) * 100)
    elif present_count > 0:
        attendance_growth = 100 # moved from 0 to something
        
    return templates.TemplateResponse("admin.html", {
        "request": request, 
        "user": user,
        "total_employees": total_employees,
        "stats": {
            "present": present_count,
            "late": late_count,
            "on_time": on_time_count,
            "absent": absent_count,
            "attendance_rate": attendance_rate,
            "weekly_total": weekly_attendance,
            "monthly_total": monthly_attendance,
            "growth": attendance_growth
        }
    })

@router.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """API endpoint to provide JSON data for dashboard charts."""
    
    today = datetime.now().date()
    
    # helper to process a date range
    def get_period_stats(days_count, period_label):
        dates = []
        counts = []
        
        # Current Period Data
        current_period_counts = []
        
        for i in range(days_count - 1, -1, -1):
            d = today - timedelta(days=i)
            # Format: Weekly uses "Mon", Monthly uses "Oct 01"
            if period_label == "weekly":
                dates.append(d.strftime("%a"))
            else:
                dates.append(d.strftime("%b %d"))
                
            d_str = d.isoformat()
            records = attendance_db.get_records_by_date(d_str)
            count = len(records)
            counts.append(count)
            current_period_counts.append((d.strftime("%A"), count)) # Store day name for peak calc
            
        # Previous Period Data (for trend)
        prev_period_total = 0
        for i in range(days_count * 2 - 1, days_count - 1, -1):
            d = today - timedelta(days=i)
            records = attendance_db.get_records_by_date(d.isoformat())
            prev_period_total += len(records)
            
        # Metrics Calculation
        current_total = sum(counts)
        avg = round(current_total / days_count, 1) if days_count > 0 else 0
        
        # Trend %
        trend = 0
        if prev_period_total > 0:
            change = current_total - prev_period_total
            trend = round((change / prev_period_total) * 100, 1)
        elif current_total > 0:
            trend = 100 # 100% growth from 0
            
        # Peak Day
        peak_day = "N/A"
        if current_period_counts:
            # Find max count
            max_val = max(counts)
            if max_val > 0:
                # Find first day with max_val
                for name, val in current_period_counts:
                    if val == max_val:
                        peak_day = name
                        break
                        
        return {
            "labels": dates,
            "data": counts,
            "statistics": {
                "average": avg,
                "total": current_total,
                "trend": trend,
                "peak_day": peak_day
            }
        }

    # 1. Weekly Stats (Current Week: Mon -> Sun)
    # weekly_stats = get_period_stats(7, "weekly") 
    # Custom logic for standard week
    weekly_dates = []
    weekly_counts = []
    weekly_period_counts = []
    
    start_of_week = today - timedelta(days=today.weekday()) # Monday
    
    for i in range(7):
        d = start_of_week + timedelta(days=i)
        weekly_dates.append(d.strftime("%a")) # Mon, Tue...
        
        # Only fetch data if d <= today (future days are 0)
        # Actually user might want to see 0 for future days, which is fine
        d_str = d.isoformat()
        records = attendance_db.get_records_by_date(d_str)
        count = len(records)
        weekly_counts.append(count)
        weekly_period_counts.append((d.strftime("%A"), count))
        
    # Calculate stats for the partial/full week
    weekly_total = sum(weekly_counts)
    # Average based on days passed so far in the week (or 7? usually 7 for standard view, or days passed)
    # Let's use 7 to keep the chart scale consistent, or maybe just days passed?
    # For now simply:
    weekly_avg = round(weekly_total / 7, 1)
    
    weekly_stats = {
        "labels": weekly_dates,
        "data": weekly_counts,
        "statistics": {
            "average": weekly_avg,
            "total": weekly_total,
            "trend": 0, # Complex to calc trend vs last week matching days, setting 0 for now or simple diff
            "peak_day": "N/A"
        }
    }
    
    # Calculate Peak Day
    if weekly_period_counts:
        max_val = max(weekly_counts)
        if max_val > 0:
            for name, val in weekly_period_counts:
                if val == max_val:
                    weekly_stats["statistics"]["peak_day"] = name
                    break
    
    # 2. Monthly Stats (Last 30 Days)
    monthly_stats = get_period_stats(30, "monthly")
        
    # 3. Department Distribution (Existing)
    users = user_db.get_all_users()
    dept_counts = {}
    for uid, udata in users.items():
        dept = udata.get("department", "Unassigned") or "Unassigned"
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
        
    return JSONResponse({
        "stats": {
            "weekly": weekly_stats,
            "monthly": monthly_stats
        },
        "distribution": {
            "labels": list(dept_counts.keys()),
            "data": list(dept_counts.values())
        }
    })

@router.get("/api/users/{name}/attendance-stats")
async def get_user_attendance_stats(name: str):
    """API endpoint for individual employee attendance statistics."""
    today = datetime.now().date()
    
    # Weekly stats (Current Week: Mon -> Sun)
    weekly_labels = []
    weekly_hours = []
    
    start_of_week = today - timedelta(days=today.weekday()) # Monday
    
    for i in range(7):
        d = start_of_week + timedelta(days=i)
        d_str = d.isoformat()
        weekly_labels.append(d.strftime("%a"))  # Mon, Tue, etc.
        
        records = attendance_db.get_records_by_date(d_str)
        user_record = records.get(name)
        
        if user_record:
            check_in = user_record.get("check_in_time")
            check_out = user_record.get("check_out_time")
            
            hours = 0
            if check_in and check_out:
                try:
                    start = datetime.fromisoformat(check_in)
                    end = datetime.fromisoformat(check_out)
                    duration = end - start
                    seconds = duration.total_seconds()
                    hours = round(seconds / 3600, 1)
                except ValueError:
                    pass
            elif check_in and not check_out and d == today:
                 # If currently checked in today, calculate approximate hours until now
                 try:
                    start = datetime.fromisoformat(check_in)
                    now = datetime.now()
                    duration = now - start
                    seconds = duration.total_seconds()
                    hours = round(seconds / 3600, 1)
                 except ValueError:
                    pass

            weekly_hours.append(hours)
        else:
            weekly_hours.append(0)
    
    # Monthly stats (last 4 weeks)
    monthly_labels = ["Week 1", "Week 2", "Week 3", "Week 4"]
    monthly_hours = []
    
    for week in range(4):
        # Calculate week range (reverse order, Week 4 is current week, Week 1 is 3 weeks ago)
        # Actually standard charts usually show Week 1 as oldest. Let's fix loop.
        # Let's verify requirements. User wants monthly stats.
        # Standard: Week 1 (Oldest) -> Week 4 (Newest/Current)
        
        week_end_date = today - timedelta(days=(3 - week) * 7) 
        week_start_date = week_end_date - timedelta(days=6)
        
        week_total = 0
        
        for day in range(7):
            d = week_start_date + timedelta(days=day)
            d_str = d.isoformat()
            records = attendance_db.get_records_by_date(d_str)
            user_record = records.get(name)
            
            if user_record:
                check_in = user_record.get("check_in_time")
                check_out = user_record.get("check_out_time")
                
                hours = 0
                if check_in and check_out:
                    try:
                        start = datetime.fromisoformat(check_in)
                        end = datetime.fromisoformat(check_out)
                        seconds = (end - start).total_seconds()
                        hours = seconds / 3600
                    except ValueError: pass
                elif check_in and not check_out and d == today:
                     # Calculate pending hours for today
                     try:
                        start = datetime.fromisoformat(check_in)
                        seconds = (datetime.now() - start).total_seconds()
                        hours = seconds / 3600
                     except ValueError: pass

                week_total += hours
        
        monthly_hours.append(round(week_total, 1))
    
    # Summary stats
    total_days_present = sum(1 for h in weekly_hours if h > 0)
    total_weekly_hours = sum(weekly_hours)
    total_monthly_hours = sum(monthly_hours)
    
    return JSONResponse({
        "weekly": {
            "labels": weekly_labels,
            "data": weekly_hours
        },
        "monthly": {
            "labels": monthly_labels,
            "data": monthly_hours
        },
        "summary": {
            "days_present_this_week": total_days_present,
            "total_weekly_hours": round(total_weekly_hours, 1),
            "total_monthly_hours": round(total_monthly_hours, 1)
        }
    })

@router.websocket("/ws/video_input")
async def video_input(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive bytes from client
            data = await websocket.receive_bytes()
            # Process frame
            camera_service.process_input_frame(data)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Video input error: {e}")

@router.get("/video_feed")
async def video_feed(mode: str = "verification"):
    return StreamingResponse(generate_frames(mode), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/register")
async def register_page(request: Request):
    if not get_current_admin(request):
        return RedirectResponse(url="/login", status_code=303)
        
    return templates.TemplateResponse("register.html", {"request": request})


@router.get("/api/registration-state")
async def get_registration_state():
    """Get current registration state for frontend polling."""
    return JSONResponse(content={
        "name": registration_state["name"],
        "stage": registration_state["stage"]
    })

# Temporary storage for captured face images (per session/name)
captured_faces_temp = {}

@router.post("/api/capture-faces")
async def capture_faces(name: str = Form(...)):
    """Step 1: Capture 5 face poses and store temporarily"""
    images = []
    
    global registration_state
    registration_state["name"] = name
    registration_state["stage"] = 0  # Start with Frontal
    
    # Configuration for pose capture
    HOLD_TIME_REQUIRED = 1.5  # Seconds face must be held in position
    DELAY_BETWEEN_POSES = 2.0  # Seconds to wait between captures
    TOTAL_TIMEOUT = 60  # Total seconds allowed for all 5 poses
    
    start_time = time.time()
    hold_start_time = None
    
    # Capture 5 poses
    while len(images) < 5:
        if time.time() - start_time > TOTAL_TIMEOUT:
            break
            
        frame = camera_service.get_frame()
        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            await asyncio.sleep(0.05)
            continue
            
        detections = detector.detect(frame)
        
        h, w = frame.shape[:2]
        bracket_w, bracket_h = int(w * 0.45), int(h * 0.7)
        bx1, by1 = (w - bracket_w) // 2, (h - bracket_h) // 2
        bx2, by2 = bx1 + bracket_w, by1 + bracket_h
        
        face_valid = False
        for (x1, y1, x2, y2), conf in detections:
            fx, fy = (x1 + x2) // 2, (y1 + y2) // 2
            if bx1 < fx < bx2 and by1 < fy < by2:
                face_w = x2 - x1
                if face_w > bracket_w * 0.3:
                    face_valid = True
                    break
        
        if face_valid:
            if hold_start_time is None:
                hold_start_time = time.time()
                registration_state["hold_progress"] = 0.0
            else:
                hold_duration = time.time() - hold_start_time
                registration_state["hold_progress"] = min(hold_duration / HOLD_TIME_REQUIRED, 1.0)
                
                if hold_duration >= HOLD_TIME_REQUIRED:
                    images.append(frame.copy())
                    print(f"Captured pose {len(images)}/5 for {name}")
                    
                    registration_state["stage"] = len(images)
                    registration_state["hold_progress"] = 0.0
                    hold_start_time = None
                    
                    if len(images) < 5:
                        await asyncio.sleep(DELAY_BETWEEN_POSES)
        else:
            hold_start_time = None
            registration_state["hold_progress"] = 0.0
        
        await asyncio.sleep(0.05)

    # Reset state
    registration_state["stage"] = -1
    registration_state["name"] = ""
    registration_state["hold_progress"] = 0.0

    if len(images) < 5:
        return JSONResponse(status_code=400, content={
            "success": False,
            "message": f"Could only capture {len(images)}/5 images. Please try again."
        })
    
    # Store captured images temporarily
    captured_faces_temp[name] = images
    
    return JSONResponse(content={
        "success": True,
        "message": "All 5 poses captured successfully!",
        "poses_captured": 5
    })


@router.post("/api/complete-registration")
async def complete_registration(
    name: str = Form(...),
    full_name: str = Form(""),
    phone: str = Form(""),
    department: str = Form(""),
    position: str = Form("")
):
    """Step 2: Complete registration with captured images and user details"""
    
    # Get captured images
    if name not in captured_faces_temp:
        return JSONResponse(status_code=400, content={
            "success": False,
            "message": "No captured faces found. Please capture face images first."
        })
    
    images = captured_faces_temp[name]
    
    try:
        success = await asyncio.to_thread(recognizer.register_user, name, images)
        if success:
            # Save user profile data to database
            user_db.create_user(name, {
                "full_name": full_name,
                "phone": phone,
                "department": department,
                "position": position
            })
            
            # Reset tracker "Unknown" counters so it re-verifies this person immediately
            face_tracker.reset_unknown_verifications()
            
            # Clean up temp storage
            del captured_faces_temp[name]
            return JSONResponse(content={
                "success": True,
                "message": f"User {name} registered successfully!"
            })
        else:
            return JSONResponse(status_code=500, content={
                "success": False,
                "message": "Registration failed during processing (Face not clear?)."
            })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": f"Error: {str(e)}"
        })

@router.get("/users")
async def list_users(request: Request):
    if not get_current_admin(request):
        return RedirectResponse(url="/login", status_code=303)
        
    # Get UNIQUE user names from recognizer (not flat list which has duplicates)
    unique_names = list(recognizer.user_encodings.keys())
    # Enrich with user data from database
    users_data = []
    for name in unique_names:
        user_info = user_db.get_user(name) or {}
        users_data.append({
            "id": name,
            "full_name": user_info.get("full_name", name),
            "phone": user_info.get("phone", ""),
            "department": user_info.get("department", ""),
            "position": user_info.get("position", "")
        })
    return templates.TemplateResponse("users.html", {"request": request, "users": users_data})

@router.get("/users/{name}")
async def user_detail(request: Request, name: str):
    if not get_current_admin(request):
        return RedirectResponse(url="/login", status_code=303)

    user_dir = os.path.join(settings.IMAGES_DIR, name)
    images = []
    if os.path.exists(user_dir):
        images = [img for img in os.listdir(user_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Get user profile data
    user_info = user_db.get_user(name) or {}
    
    return templates.TemplateResponse("user_detail.html", {
        "request": request, 
        "name": name, 
        "images": sorted(images),
        "user_info": user_info
    })

@router.get("/users/{name}/edit")
async def edit_user_page(request: Request, name: str):
    user_info = user_db.get_user(name) or {}
    return templates.TemplateResponse("user_edit.html", {
        "request": request,
        "name": name,
        "user_info": user_info
    })

@router.post("/api/users/{name}/update")
async def update_user(
    name: str,
    full_name: str = Form(""),
    phone: str = Form(""),
    department: str = Form(""),
    position: str = Form("")
):
    success = user_db.update_user(name, {
        "full_name": full_name,
        "phone": phone,
        "department": department,
        "position": position
    })
    if success:
        return JSONResponse(content={"message": f"User {name} updated successfully."})
    else:
        # User doesn't exist in DB yet, create entry
        user_db.create_user(name, {
            "full_name": full_name,
            "phone": phone,
            "department": department,
            "position": position
        })
        return JSONResponse(content={"message": f"User {name} profile created."})

@router.delete("/api/users/{name}")
async def delete_user(name: str):
    import shutil
    # Delete from recognizer (remove ALL encodings for this user)
    if name in recognizer.user_encodings:
        del recognizer.user_encodings[name]
        recognizer._rebuild_flat_lists()  # Rebuild flat lists from user_encodings
        recognizer.save_encodings()
        
    # Remove from active tracker (so they don't stay recognized)
    face_tracker.remove_user(name)
    
    # Delete from user database
    user_db.delete_user(name)
    
    # Delete images folder
    user_dir = os.path.join(settings.IMAGES_DIR, name)
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
    
    return JSONResponse(content={"message": f"User {name} deleted successfully."})


# ========== WEBSOCKET ENDPOINT ==========

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time attendance notifications."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, receive any client messages
            data = await websocket.receive_text()
            # Could handle client messages here if needed
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)


# ========== ATTENDANCE PROCESSING ==========

async def process_attendance_queue():
    """Background task to process attendance events and broadcast via WebSocket."""
    while True:
        try:
            # Wait for an event from the queue
            event = await attendance_queue.get()
            worker_id = event["worker_id"]
            frame = event["frame"]
            log_db = event.get("log_db", True)
            
            if worker_id == "Unknown":
                # Special handling for unknown users
                await ws_manager.send_check_in("Unknown", "Unknown")
            
            elif log_db:
                # Record attendance (heavy db write)
                result = await asyncio.to_thread(
                    attendance_db.record_attendance, worker_id, frame
                )
            else:
                # Read-only: Get current status for visual feedback
                record = attendance_db.get_today_record(worker_id)
                if record:
                    # Determine event type based on existing record state
                    if record.get("check_out_time"):
                        event_type = "check_out"
                    else:
                        event_type = "check_in"
                    result = {"event_type": event_type, "record": record}
                else:
                    # Should unlikely happen if visual debounce > log debounce
                    # Fallback
                    result = {
                        "event_type": "check_in", 
                        "record": {"check_in_time": datetime.now().isoformat(), "check_out_time": None}
                    }
            
            # Process Known Users
            if worker_id != "Unknown":
                # Get user info for display name
                user_info = user_db.get_user(worker_id)
                full_name = user_info.get("full_name", worker_id) if user_info else worker_id
                
                # Broadcast event via WebSocket
                if result["event_type"] == "check_in":
                    await ws_manager.send_check_in(worker_id, full_name)
                else:
                    # Calculate working time
                    working_time = "N/A"
                    check_in_str = None
                    try:
                        cin = datetime.fromisoformat(result["record"]["check_in_time"])
                        # Use checkout time if available, otherwise use now (just for diff, though usually cout exists here)
                        cout_str = result["record"].get("check_out_time")
                        cout = datetime.fromisoformat(cout_str) if cout_str else datetime.now()
                        
                        # Format Check In Time for display (e.g., 09:30 AM)
                        check_in_str = cin.strftime("%I:%M %p")
                        
                        duration = cout - cin
                        hours, remainder = divmod(duration.seconds, 3600)
                        minutes, _ = divmod(remainder, 60)
                        working_time = f"{hours}h {minutes}m"
                    except Exception as e:
                        print(f"Error calc duration: {e}")
                        
                    await ws_manager.send_check_out(worker_id, full_name, working_time, check_in_time=check_in_str)
                
        except Exception as e:
            print(f"Attendance processing error: {e}")
        
        # Immediate processing of next item
        await asyncio.sleep(0)


# Start the background task when the module loads
_attendance_task = None

def start_attendance_processor():
    """Start the attendance processing background task."""
    global _attendance_task
    if _attendance_task is None:
        loop = asyncio.get_event_loop()
        _attendance_task = loop.create_task(process_attendance_queue())


# ========== ATTENDANCE API ENDPOINTS ==========

@router.get("/attendance")
async def attendance_page(request: Request):
    """Serve the attendance dashboard page."""
    if not get_current_admin(request):
        return RedirectResponse(url="/login", status_code=303)
        
    all_records = attendance_db.get_all_records()
    
    # Enrich records with user details and calculations
    display_records = []
    for record in all_records:
        worker_id = record["worker_id"]
        user_info = user_db.get_user(worker_id)
        is_known = (user_info is not None)
        
        # Calculate Working Hours
        working_hours = "-"
        if record.get("check_in_time") and record.get("check_out_time"):
            try:
                cin = datetime.fromisoformat(record["check_in_time"])
                cout = datetime.fromisoformat(record["check_out_time"])
                dt = cout - cin
                h, r = divmod(dt.seconds, 3600)
                m, _ = divmod(r, 60)
                working_hours = f"{h}h {m}m"
            except:
                pass
        
        display_records.append({
            "date": record.get("date"),
            "worker_id": worker_id,
            "full_name": user_info.get("full_name", worker_id) if user_info else worker_id,
            "department": user_info.get("department", "Unknown") if user_info else "Unknown",
            "check_in_time": record.get("check_in_time"),
            "check_out_time": record.get("check_out_time"),
            "check_in_snapshot": record.get("check_in_snapshot"),
            "check_out_snapshot": record.get("check_out_snapshot"),
            "working_hours": working_hours,
            "is_known": is_known
        })
    
    # Sort by check-in time descending
    display_records.sort(key=lambda x: x["check_in_time"] or "", reverse=True)
    
    return templates.TemplateResponse("attendance.html", {
        "request": request, 
        "attendance_records": display_records
    })

@router.delete("/api/attendance/{date_str}/{worker_id}")
async def delete_attendance_record(date_str: str, worker_id: str):
    """Delete a specific attendance record."""
    if attendance_db.delete_record(date_str, worker_id):
        return JSONResponse({"success": True})
    return JSONResponse({"success": False, "message": "Record not found"}, status_code=404)

@router.get("/api/attendance/all")
async def get_all_attendance(
    start_date: str = None, 
    end_date: str = None
):
    """Get all attendance records enriched with user details."""
    raw_records = attendance_db.get_all_records(start_date, end_date)
    enriched_records = []
    
    for record in raw_records:
        worker_id = record["worker_id"]
        user_info = user_db.get_user(worker_id) or {}
        
        # Calculate Working Hours
        working_hours = "-"
        if record.get("check_in_time") and record.get("check_out_time"):
            try:
                cin = datetime.fromisoformat(record["check_in_time"])
                cout = datetime.fromisoformat(record["check_out_time"])
                dt = cout - cin
                h, r = divmod(dt.seconds, 3600)
                m, _ = divmod(r, 60)
                working_hours = f"{h}h {m}m"
            except:
                pass

        # Merge record with user info
        enriched_record = {
            **record,
            "full_name": user_info.get("full_name", worker_id),
            "department": user_info.get("department", "Unknown"),
            "position": user_info.get("position", "Unknown"),
            # Format times for display if present
            "check_in_display": record["check_in_time"].split('T')[1][:8] if record["check_in_time"] else "-",
            "check_out_display": record["check_out_time"].split('T')[1][:8] if record["check_out_time"] else "-",
            "working_hours": working_hours
        }
        enriched_records.append(enriched_record)
        
    return JSONResponse(content=enriched_records)


@router.get("/api/attendance/today")
async def get_today_attendance():
    """Get all attendance records for today."""
    records = attendance_db.get_all_today()
    return JSONResponse(content={"date": attendance_db._get_today_key(), "records": records})


@router.get("/api/attendance/{worker_id}")
async def get_worker_attendance(worker_id: str, limit: int = 30):
    """Get attendance history for a specific worker."""
    history = attendance_db.get_worker_history(worker_id, limit)
    return JSONResponse(content={"worker_id": worker_id, "history": history})
