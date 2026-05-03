from fastapi import APIRouter, Request, Form, WebSocket, WebSocketDisconnect, Body
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse, Response
from app.services.camera import camera_service
from app.services.detector import detector
from app.services.recognizer import recognizer
import cv2
import numpy as np
import time
from datetime import datetime, timedelta, date as date_type
from io import BytesIO
import asyncio
import os
from app.core.config import settings

from app.services.tracker import Tracker
from app.services.user_db import user_db
from app.services.departments_db import departments_db
from app.services.attendance_db import attendance_db
from app.services.websocket_manager import ws_manager
from app.services.schedule_settings import (
    load_schedule,
    save_schedule,
    classify_attendance_status,
    work_bounds_for_date,
)
from app.services.holidays_db import holidays_db
from app.services import audit_log
from app.services.telegram_notify import (
    send_telegram_message,
    send_telegram_message_result,
    build_daily_summary_text,
    fetch_chat_ids_from_updates,
    save_chat_id_to_file,
    is_telegram_ready,
    get_effective_chat_id,
    process_telegram_updates_long_poll,
    notify_attendance_event,
    get_extra_chat_ids,
    save_extra_chat_ids,
    send_daily_summary_with_chart,
)
from app.services.video_processor import get_or_create_processor, remove_processor
import base64
import uuid

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Attendance event queue for async processing
attendance_queue = asyncio.Queue()

# Debounce tracking: {worker_id: last_logged_time}
attendance_debounce = {}
visual_debounce = {}
DEBOUNCE_SECONDS = 30  # Only log to DB once per 30 seconds
VISUAL_DEBOUNCE_SECONDS = 3  # Show visual feedback every 3 seconds
pending_snapshots = {} # {tid: frame_copy}

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
                # ── Face quality filter ──
                # Skip faces that are too small or have low detection confidence
                MIN_FACE_SIZE = 80   # pixels
                MIN_CONFIDENCE = 0.65
                
                bbox_list = []
                for det in detections:
                    # det is ((x1, y1, x2, y2), conf)
                    bbox, conf = det
                    x1d, y1d, x2d, y2d = bbox
                    face_w = x2d - x1d
                    face_h = y2d - y1d
                    if face_w >= MIN_FACE_SIZE and face_h >= MIN_FACE_SIZE and conf >= MIN_CONFIDENCE:
                        bbox_list.append(bbox)
                
                tracked_faces = face_tracker.update(bbox_list)
                
                # Cleanup old snapshots
                global pending_snapshots
                active_tids = set(f["id"] for f in tracked_faces)
                pending_snapshots = {k: v for k, v in pending_snapshots.items() if k in active_tids}
                
                for face in tracked_faces:
                    tid = face["id"]
                    x1, y1, x2, y2 = face["bbox"]
                    name = face["name"]
                    needs_reverify = face.get("needs_reverify", False)
                    is_confirmed = face.get("is_confirmed", False)
                    
                    # Recognition: if name not cached OR needs re-verification
                    if name is None or needs_reverify:
                        try:
                            name = recognizer.verify(frame, (x1, y1, x2, y2))
                            face_tracker.set_name(tid, name)
                            # Re-read confirmation status after voting
                            is_confirmed = face_tracker.tracks.get(tid, {}).get("confirmed", False)
                            
                            # Cache the frame on the first successful vote
                            pending_votes = face_tracker.tracks.get(tid, {}).get("pending_votes", 0)
                            if pending_votes == 1 and name != "Unknown":
                                pending_snapshots[tid] = frame.copy()
                                
                        except Exception:
                            name = "Unknown"
                            face_tracker.set_name(tid, name)
                    
                    # ── Attendance logging & Visual Feedback ──
                    # CRITICAL: Only log to DB when identity is CONFIRMED (multi-frame voting passed)
                    if name:
                        current_time = time.time()
                        last_db = attendance_debounce.get(name, 0)
                        last_visual = visual_debounce.get(name, 0)
                        
                        # Logic for Known Users
                        if name != "Unknown":
                            # Only log attendance for CONFIRMED identities
                            if is_confirmed:
                                should_log_db = (current_time - last_db >= DEBOUNCE_SECONDS)
                                should_show_visual = (current_time - last_visual >= VISUAL_DEBOUNCE_SECONDS)
                                
                                if should_show_visual:
                                    visual_debounce[name] = current_time
                                    if should_log_db:
                                        attendance_debounce[name] = current_time
                                    
                                    try:
                                        snapshot = pending_snapshots.get(tid, frame).copy()
                                        attendance_queue.put_nowait({
                                            "worker_id": name,
                                            "frame": snapshot,
                                            "log_db": should_log_db
                                        })
                                    except asyncio.QueueFull:
                                        pass
                            # else: not confirmed yet — do NOT log to DB
                        
                        # Logic for Unknown Users
                        else:
                            should_show_visual = (current_time - last_visual >= VISUAL_DEBOUNCE_SECONDS)
                            
                            if should_show_visual:
                                visual_debounce[name] = current_time
                                try:
                                    attendance_queue.put_nowait({
                                        "worker_id": "Unknown",
                                        "frame": frame.copy(),
                                        "log_db": False
                                    })
                                except asyncio.QueueFull:
                                    pass
                    
                    # ── Draw Results ──
                    # 3 states: Confirmed (green), Verifying (yellow), Unknown (red)
                    if name:
                        if name == "Unknown":
                            color = (0, 0, 255)  # Red
                            label = "Unknown"
                            text_color = (255, 255, 255)  # White text
                        elif is_confirmed:
                            color = (0, 255, 0)  # Green — confirmed
                            label = f"ID:{tid} | {name}"
                            text_color = (0, 0, 0)  # Black text
                        else:
                            color = (0, 200, 255)  # Yellow/Orange — verifying
                            label = f"Verifying: {name}..."
                            text_color = (0, 0, 0)  # Black text

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
    
    username = active_sessions[session_id]
    # Fetch full user details (mocking verification again or just trusting session)
    # Ideally we should store user role in session or fetch from DB. 
    # For now, let's fetch from DB to be safe and get the role.
    from app.services.admin_db import _load_admins # Or expose a get_user method
    admins = _load_admins()
    user = admins.get(username)
    if user and "role" not in user:
        user["role"] = "admin"
    return user

TELEGRAM_SENT_FLAG = os.path.join(settings.DATA_DIR, "telegram_last_sent.txt")


async def telegram_daily_scheduler_loop():
    """Kunlik Telegram xulosasini sozlangan vaqtda bir marta yuborish (grafik bilan)."""
    while True:
        await asyncio.sleep(45)
        try:
            if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
                continue
            now = datetime.now()
            if (
                now.hour != settings.TELEGRAM_DAILY_SUMMARY_HOUR
                or now.minute != settings.TELEGRAM_DAILY_SUMMARY_MINUTE
            ):
                continue
            today_s = now.date().isoformat()
            if os.path.exists(TELEGRAM_SENT_FLAG):
                with open(TELEGRAM_SENT_FLAG, "r", encoding="utf-8") as f:
                    if f.read().strip() == today_s:
                        continue
            result = await asyncio.to_thread(send_daily_summary_with_chart)
            if result.get("ok"):
                os.makedirs(settings.DATA_DIR, exist_ok=True)
                with open(TELEGRAM_SENT_FLAG, "w", encoding="utf-8") as f:
                    f.write(today_s)
        except Exception as e:
            print(f"Telegram scheduler: {e}")


async def telegram_bot_updates_loop():
    """Long polling: /start ga javob, Chat ID ko'rsatish (webhook bo'lmasa)."""
    while True:
        if not (settings.TELEGRAM_BOT_TOKEN or "").strip():
            await asyncio.sleep(60)
            continue
        try:
            await asyncio.to_thread(process_telegram_updates_long_poll)
        except Exception as e:
            print(f"Telegram bot polling: {e}")
        await asyncio.sleep(3)  # Give event loop breathing room between polls


@router.get("/")
async def index(request: Request):
    user = get_current_admin(request)
    if not user:
        return RedirectResponse(url="/login?next=/", status_code=303)
        
    # Strict separation: Admin users cannot access Kiosk dashboard
    if user.get("role") == "admin":
        return RedirectResponse(url="/admin", status_code=303)

        
    active_users = len(recognizer.get_all_user_names())
    return templates.TemplateResponse(request=request, name="index.html", context= {
        "request": request,
        "active_users_count": active_users,
        "user": user
    })

@router.get("/login")
async def login_page(request: Request, next: str = "/admin"):
    return templates.TemplateResponse(request=request, name="login.html", context= {"request": request, "next": next})

@router.post("/login")
async def login_action(request: Request, username: str = Form(...), password: str = Form(...), next: str = "/admin"):
    user = admin_db.verify_admin(username, password)
    if user:
        # Create session
        token = secrets.token_hex(16)
        active_sessions[token] = username
        
        # Role-based redirection
        role = user.get("role", "admin")
        
        if role == "kiosk":
            redirect_url = "/" # Kiosk users ALWAYS go to kiosk page
        else:
            # Admins go to requested page or default admin dashboard
            redirect_url = next if next.startswith("/") else "/admin"
        
        response = RedirectResponse(url=redirect_url, status_code=303)
        response.set_cookie(key="admin_session", value=token, httponly=True)
        return response
    else:
        return templates.TemplateResponse(request=request, name="login.html", context= {
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

@router.get("/daily_report")
async def daily_report(request: Request):
    user = get_current_admin(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    
    if user.get("role") != "admin":
        return RedirectResponse(url="/login?next=/daily_report", status_code=303)

    users = user_db.get_all_users()
    today_records = attendance_db.get_all_today()
    today_d = datetime.now().date()
    today_iso = today_d.isoformat()
    holiday_label = holidays_db.get_label(today_iso)

    present_users = []
    absent_users = []

    work_start_dt, work_end_dt = work_bounds_for_date(today_d)

    for name, user_data in users.items():
        if name in today_records:
            record = today_records[name]
            check_in_str = record.get("check_in_time", "")
            check_out_str = record.get("check_out_time", "")
            check_in_dt = None
            check_out_dt = None

            if check_in_str:
                try:
                    check_in_dt = datetime.fromisoformat(check_in_str)
                    check_in_str = check_in_dt.strftime("%H:%M")
                except ValueError:
                    pass

            if check_out_str:
                try:
                    check_out_dt = datetime.fromisoformat(check_out_str)
                    check_out_str = check_out_dt.strftime("%H:%M")
                except ValueError:
                    pass

            st = classify_attendance_status(check_in_dt, check_out_dt, today_d)
            status_label = st["label"]
            status_type = st["type"]

            present_users.append({
                "username": name,
                "full_name": user_data.get("full_name", name),
                "department": user_data.get("department", "Unassigned"),
                "check_in_time": check_in_str,
                "check_out_time": check_out_str,
                "status_label": status_label,
                "status_type": status_type
            })
        else:
            if not holiday_label:
                absent_users.append({
                    "username": name,
                    "full_name": user_data.get("full_name", name),
                    "department": user_data.get("department", "Unassigned")
                })

    return templates.TemplateResponse(request=request, name="daily_report.html", context= {
        "request": request,
        "user": user,
        "present_users": present_users,
        "absent_users": absent_users,
        "today_date": today_iso,
        "holiday_label": holiday_label,
        "work_start": work_start_dt.strftime("%H:%M"),
        "work_end": work_end_dt.strftime("%H:%M"),
    })


@router.get("/api/daily-report/export/excel")
async def export_daily_report_excel(request: Request):
    """Kunlik hisobotni Excel (.xlsx) fayl sifatida yuklab olish."""
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    users = user_db.get_all_users()
    today_records = attendance_db.get_all_today()
    today_d = datetime.now().date()
    today_iso = today_d.isoformat()
    holiday_label = holidays_db.get_label(today_iso)

    work_start_dt, work_end_dt = work_bounds_for_date(today_d)

    present_users = []
    absent_users = []

    for name, user_data in users.items():
        if name in today_records:
            record = today_records[name]
            check_in_str = record.get("check_in_time", "")
            check_out_str = record.get("check_out_time", "")
            check_in_dt = None
            check_out_dt = None

            if check_in_str:
                try:
                    check_in_dt = datetime.fromisoformat(check_in_str)
                    check_in_str = check_in_dt.strftime("%H:%M")
                except ValueError:
                    pass

            if check_out_str:
                try:
                    check_out_dt = datetime.fromisoformat(check_out_str)
                    check_out_str = check_out_dt.strftime("%H:%M")
                except ValueError:
                    pass

            st = classify_attendance_status(check_in_dt, check_out_dt, today_d)

            present_users.append({
                "username": name,
                "full_name": user_data.get("full_name", name),
                "department": user_data.get("department", "Unassigned"),
                "check_in_time": check_in_str,
                "check_out_time": check_out_str,
                "status_label": st["label"],
            })
        else:
            if not holiday_label:
                absent_users.append({
                    "username": name,
                    "full_name": user_data.get("full_name", name),
                    "department": user_data.get("department", "Unassigned"),
                })

    wb = Workbook()
    ws = wb.active
    ws.title = "Kunlik hisobot"

    # Title row
    ws.merge_cells("A1:F1")
    title_cell = ws["A1"]
    title_cell.value = f"Kunlik hisobot — {today_iso} (Grafik: {work_start_dt.strftime('%H:%M')} – {work_end_dt.strftime('%H:%M')})"
    title_cell.font = Font(bold=True, size=14)
    title_cell.alignment = Alignment(horizontal="center")

    # Present users section
    row = 3
    ws.cell(row=row, column=1, value="KELGANLAR").font = Font(bold=True, size=12, color="166534")
    row += 1

    header_fill = PatternFill(start_color="E2E8F0", end_color="E2E8F0", fill_type="solid")
    headers = ["#", "Xodim ID", "To'liq ism", "Bo'lim", "Kelgan vaqti", "Holati"]
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=row, column=col, value=h)
        cell.font = Font(bold=True, size=10)
        cell.fill = header_fill
    row += 1

    for i, p in enumerate(present_users, 1):
        ws.cell(row=row, column=1, value=i)
        ws.cell(row=row, column=2, value=p["username"])
        ws.cell(row=row, column=3, value=p["full_name"])
        ws.cell(row=row, column=4, value=p["department"])
        ws.cell(row=row, column=5, value=p["check_in_time"])
        ws.cell(row=row, column=6, value=p["status_label"])
        row += 1

    # Absent users section
    row += 1
    ws.cell(row=row, column=1, value="KELMAGANLAR").font = Font(bold=True, size=12, color="B91C1C")
    row += 1

    absent_headers = ["#", "Xodim ID", "To'liq ism", "Bo'lim", "Holati"]
    for col, h in enumerate(absent_headers, 1):
        cell = ws.cell(row=row, column=col, value=h)
        cell.font = Font(bold=True, size=10)
        cell.fill = header_fill
    row += 1

    for i, a in enumerate(absent_users, 1):
        ws.cell(row=row, column=1, value=i)
        ws.cell(row=row, column=2, value=a["username"])
        ws.cell(row=row, column=3, value=a["full_name"])
        ws.cell(row=row, column=4, value=a["department"])
        ws.cell(row=row, column=5, value="Kelmagan")
        row += 1

    # Auto-width columns
    for col in ws.columns:
        max_length = 0
        column_letter = None
        for cell in col:
            if hasattr(cell, 'column_letter'):
                column_letter = cell.column_letter
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        if column_letter:
            ws.column_dimensions[column_letter].width = min(max_length + 4, 40)

    buf = BytesIO()
    wb.save(buf)

    filename = f"kunlik_hisobot_{today_iso}.xlsx"
    return Response(
        content=buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/admin")
async def admin_dashboard(request: Request):
    user = get_current_admin(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    
    # Check if user has admin role
    if user.get("role") != "admin":
        return RedirectResponse(url="/login?next=/admin", status_code=303)

    
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
        
        ws, _we = work_bounds_for_date(today_date)
        for name, record in today_records.items():
            check_in_str = record.get("check_in_time")
            if check_in_str:
                try:
                    check_in_dt = datetime.fromisoformat(check_in_str)
                    if check_in_dt > ws:
                        late_count += 1
                    else:
                        on_time_count += 1
                except ValueError:
                    pass

        if holidays_db.is_holiday(today_date.isoformat()):
            absent_count = 0
        else:
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
        
    return templates.TemplateResponse(request=request, name="admin.html", context= {
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
    
    # Yearly stats (Jan -> Dec)
    import calendar
    yearly_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    yearly_hours = [0] * 12
    current_year = today.year
    
    for month_index in range(12):
        month_number = month_index + 1
        month_total = 0
        num_days = calendar.monthrange(current_year, month_number)[1]
        
        for day in range(1, num_days + 1):
            try:
                d = datetime(current_year, month_number, day).date()
            except ValueError:
                continue
            if d > today:
                break
                
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

                month_total += hours
                
        yearly_hours[month_index] = round(month_total, 1)

    # Summary stats
    total_days_present = sum(1 for h in weekly_hours if h > 0)
    total_weekly_hours = sum(weekly_hours)
    total_monthly_hours = sum(monthly_hours)
    total_yearly_hours = sum(yearly_hours)
    
    return JSONResponse({
        "weekly": {
            "labels": weekly_labels,
            "data": weekly_hours
        },
        "monthly": {
            "labels": monthly_labels,
            "data": monthly_hours
        },
        "yearly": {
            "labels": yearly_labels,
            "data": yearly_hours
        },
        "summary": {
            "days_present_this_week": total_days_present,
            "total_weekly_hours": round(total_weekly_hours, 1),
            "total_monthly_hours": round(total_monthly_hours, 1),
            "total_yearly_hours": round(total_yearly_hours, 1)
        }
    })

@router.get("/video_feed")
async def video_feed(mode: str = "verification"):
    return StreamingResponse(generate_frames(mode), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/register")
async def register_page(request: Request):
    user = get_current_admin(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    
    # Allow both admin and kiosk roles
    # if user.get("role") not in ["admin", "kiosk"]:
    #    return RedirectResponse(url="/", status_code=303)
        
    return templates.TemplateResponse(request=request, name="register.html", context= {"request": request, "departments": departments_db.get_all()})


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
        # Create user record FIRST (face_encodings has FK to users)
        user_db.create_user(name, {
            "full_name": full_name,
            "phone": phone,
            "department": department,
            "position": position
        })

        success = await asyncio.to_thread(recognizer.register_user, name, images)
        if success:
            # Reset tracker "Unknown" counters so it re-verifies this person immediately
            face_tracker.reset_unknown_verifications()
            
            # Clean up temp storage
            del captured_faces_temp[name]
            return JSONResponse(content={
                "success": True,
                "message": f"User {name} registered successfully!"
            })
        else:
            # Rollback: delete user if face encoding failed
            user_db.delete_user(name)
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
    user = get_current_admin(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    
    if user.get("role") != "admin":
        return RedirectResponse(url="/login?next=/users", status_code=303)
        
    # Get UNIQUE user names from recognizer (not flat list which has duplicates)
    unique_names = recognizer.get_all_user_names()
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
    return templates.TemplateResponse(request=request, name="users.html", context= {"request": request, "users": users_data, "departments": departments_db.get_all()})

@router.get("/users/{name}")
async def user_detail(request: Request, name: str):
    user = get_current_admin(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
        
    if user.get("role") != "admin":
        return RedirectResponse(url="/", status_code=303)

    user_dir = os.path.join(settings.IMAGES_DIR, name)
    images = []
    if os.path.exists(user_dir):
        images = [img for img in os.listdir(user_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Get user profile data
    user_info = user_db.get_user(name) or {}
    
    return templates.TemplateResponse(request=request, name="user_detail.html", context= {
        "request": request, 
        "name": name, 
        "images": sorted(images),
        "user_info": user_info
    })

@router.get("/users/{name}/edit")
async def edit_user_page(request: Request, name: str):
    user = get_current_admin(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
        
    if user.get("role") != "admin":
        return RedirectResponse(url="/", status_code=303)
        
    user_info = user_db.get_user(name) or {}
    return templates.TemplateResponse(request=request, name="user_edit.html", context= {
        "request": request,
        "name": name,
        "user_info": user_info,
        "departments": departments_db.get_all()
    })

@router.post("/api/users/{name}/update")
async def update_user(
    request: Request,
    name: str,
    full_name: str = Form(""),
    phone: str = Form(""),
    department: str = Form(""),
    position: str = Form("")
):
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse(status_code=403, content={"message": "Unauthorized"})
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
async def delete_user(request: Request, name: str):
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse(status_code=403, content={"message": "Unauthorized"})
    import shutil
    # Delete from recognizer (remove ALL encodings for this user from DB)
    recognizer.delete_user_encodings(name)
        
    # Remove from active tracker (so they don't stay recognized)
    face_tracker.remove_user(name)
    
    # Delete from user database
    user_db.delete_user(name)
    
    # Delete images folder
    user_dir = os.path.join(settings.IMAGES_DIR, name)
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
    
    return JSONResponse(content={"message": f"User {name} deleted successfully."})


# ========== BROWSER CAMERA ENDPOINTS ==========

@router.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """WebSocket endpoint for receiving browser camera frames and returning recognition results."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    processor = get_or_create_processor(session_id)
    print(f"Video WebSocket connected: {session_id}")
    
    try:
        while True:
            # Receive binary JPEG frame from browser
            data = await websocket.receive()
            
            if "bytes" in data:
                frame_bytes = data["bytes"]
            elif "text" in data:
                # Handle JSON messages (e.g., mode switch)
                import json
                msg = json.loads(data["text"])
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue
                continue
            else:
                continue
            
            # Decode frame (blocking -> run in thread)
            frame = await asyncio.to_thread(processor.decode_frame, frame_bytes)
            if frame is None:
                continue
            
            # Process for verification (blocking -> run in thread)
            result = await asyncio.to_thread(processor.process_verification_frame, frame, attendance_queue)
            
            # Send results back to browser
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        print(f"Video WebSocket disconnected: {session_id}")
        remove_processor(session_id)
    except Exception as e:
        print(f"Video WebSocket error: {e}")
        remove_processor(session_id)


@router.post("/api/capture-faces-browser")
async def capture_faces_browser(request: Request):
    """Capture face images from browser camera for registration.
    
    Accepts JSON with:
    - name: username
    - images: list of base64 encoded JPEG images (5 poses)
    """
    import json
    body = await request.json()
    name = body.get("name", "").strip()
    images_b64 = body.get("images", [])
    
    if not name:
        return JSONResponse(status_code=400, content={
            "success": False,
            "message": "Username is required."
        })
    
    if len(images_b64) < 5:
        return JSONResponse(status_code=400, content={
            "success": False,
            "message": f"Need 5 images, got {len(images_b64)}."
        })
    
    # Decode base64 images to OpenCV frames
    images = []
    for i, img_b64 in enumerate(images_b64[:5]):
        try:
            # Remove data URL prefix if present
            if "," in img_b64:
                img_b64 = img_b64.split(",")[1]
            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                # Validate face is present
                detections = detector.detect(frame)
                if detections:
                    images.append(frame)
                else:
                    print(f"No face detected in image {i} for {name}")
            else:
                print(f"Failed to decode image {i} for {name}")
        except Exception as e:
            print(f"Error processing image {i} for {name}: {e}")
    
    if len(images) < 3:
        return JSONResponse(status_code=400, content={
            "success": False,
            "message": f"Only {len(images)} valid face images detected. Need at least 3. Please try again."
        })
    
    # Store captured images temporarily (same as existing capture_faces)
    captured_faces_temp[name] = images
    
    return JSONResponse(content={
        "success": True,
        "message": f"{len(images)} face images captured successfully!",
        "poses_captured": len(images)
    })


@router.post("/api/detect-face-browser")
async def detect_face_browser(request: Request):
    """Detect face in a single frame from browser camera.
    Used during registration for real-time face position feedback.
    """
    body = await request.json()
    img_b64 = body.get("image", "")
    
    if not img_b64:
        return JSONResponse(content={"face_detected": False})
    
    try:
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JSONResponse(content={"face_detected": False})
        
        # Use a temp processor for detection
        temp_processor = get_or_create_processor("registration-detect")
        result = temp_processor.process_registration_frame(frame)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"face_detected": False, "error": str(e)})


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

                if log_db:
                    await asyncio.to_thread(
                        notify_attendance_event,
                        result["event_type"],
                        full_name,
                        worker_id,
                        result["record"],
                    )
                
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
    user = get_current_admin(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
        
    if user.get("role") != "admin":
        return RedirectResponse(url="/", status_code=303)
        
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
    
    return templates.TemplateResponse(request=request, name="attendance.html", context= {
        "request": request, 
        "attendance_records": display_records,
        "departments": departments_db.get_all()
    })

@router.delete("/api/attendance/{date_str}/{worker_id}")
async def delete_attendance_record(request: Request, date_str: str, worker_id: str):
    """Delete a specific attendance record."""
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"success": False, "message": "Forbidden"}, status_code=403)
    info = user_db.get_user(worker_id) or {}
    wname = info.get("full_name", worker_id)
    actor = user.get("username", "admin")
    if attendance_db.delete_record(date_str, worker_id):
        audit_log.log_attendance_delete(actor, date_str, worker_id, wname)
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

        working_hours = "-"
        if record.get("check_in_time") and record.get("check_out_time"):
            try:
                cin = datetime.fromisoformat(record["check_in_time"])
                cout = datetime.fromisoformat(record["check_out_time"])
                total_sec = int((cout - cin).total_seconds())
                if total_sec >= 0:
                    h, r = divmod(total_sec, 3600)
                    m, _ = divmod(r, 60)
                    working_hours = f"{h}h {m}m"
            except Exception:
                pass

        d_str = record.get("date") or ""
        try:
            rd = date_type.fromisoformat(d_str) if d_str else date_type.today()
        except ValueError:
            rd = date_type.today()
        cin_dt = (
            datetime.fromisoformat(record["check_in_time"])
            if record.get("check_in_time")
            else None
        )
        cout_dt = (
            datetime.fromisoformat(record["check_out_time"])
            if record.get("check_out_time")
            else None
        )
        st = classify_attendance_status(cin_dt, cout_dt, rd)

        enriched_record = {
            **record,
            "full_name": user_info.get("full_name", worker_id),
            "department": user_info.get("department", "Unknown"),
            "position": user_info.get("position", "Unknown"),
            "check_in_display": record["check_in_time"].split("T")[1][:8]
            if record.get("check_in_time")
            else "-",
            "check_out_display": record["check_out_time"].split("T")[1][:8]
            if record.get("check_out_time")
            else "-",
            "working_hours": working_hours,
            "status_label": st["label"],
            "late": st["late"],
            "early_leave": st["early_leave"],
            "status_type": st["type"],
        }
        enriched_records.append(enriched_record)
        
    return JSONResponse(content=enriched_records)


def _format_working_hours(check_in_iso, check_out_iso) -> str:
    if not check_in_iso or not check_out_iso:
        return "-"
    try:
        cin = datetime.fromisoformat(check_in_iso)
        cout = datetime.fromisoformat(check_out_iso)
        total_sec = int((cout - cin).total_seconds())
        if total_sec < 0:
            return "-"
        h, r = divmod(total_sec, 3600)
        m, _ = divmod(r, 60)
        return f"{h}h {m}m"
    except Exception:
        return "-"


@router.get("/api/attendance/export/excel")
async def export_attendance_excel(
    start_date: str = None,
    end_date: str = None,
):
    """Barcha davomat yozuvlarini Matrix (Kalendar) Excel (.xlsx) fayl sifatida yuklab olish."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from datetime import datetime, date, timedelta

    # Agar sana tanlanmagan bo'lsa, oxirgi 15 kunni olamiz
    if not start_date or not end_date:
        today = date.today()
        if not end_date:
            end_date = today.isoformat()
        if not start_date:
            start_date = (today - timedelta(days=14)).isoformat()

    try:
        start_dt = date.fromisoformat(start_date)
        end_dt = date.fromisoformat(end_date)
    except ValueError:
        return Response("Noto'g'ri sana formati", status_code=400)

    # Sana oraliqlarini ro'yxatga olamiz
    date_list = []
    current = start_dt
    while current <= end_dt:
        date_list.append(current)
        current += timedelta(days=1)

    # 1. Barcha xodimlar va yozuvlarni olamiz
    users = user_db.get_all_users()
    raw_records = attendance_db.get_all_records(start_date, end_date)

    # 2. Yozuvlarni xodim va sana bo'yicha guruhlaymiz: dict[worker_id][date_str] = record
    user_records = {}
    for r in raw_records:
        wid = r["worker_id"]
        d_str = r.get("date")
        if not d_str:
            continue
        if wid not in user_records:
            user_records[wid] = {}
        user_records[wid][d_str] = r

    # Excel fayl yaratamiz
    wb = Workbook()
    ws = wb.active
    ws.title = "Oylik Hisobot"

    # Uslublar (Styles)
    bold_font = Font(bold=True, size=10)
    title_font = Font(bold=True, size=14)
    center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left_align = Alignment(horizontal="left", vertical="center", wrap_text=True)
    header_fill = PatternFill(start_color="E2E8F0", end_color="E2E8F0", fill_type="solid")
    weekend_fill = PatternFill(start_color="FFDDBB", end_color="FFDDBB", fill_type="solid") # Orange
    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))

    # Sarlavha va logotip uchun qatorlar balandligini to'g'rilash
    ws.row_dimensions[1].height = 40
    ws.row_dimensions[2].height = 40
    ws.row_dimensions[3].height = 30

    # Logotip uchun joy ajratish (ba'zi Excel dasturlari rasmni katakka moslab qisqartirmasligi uchun)
    ws.merge_cells("A1:E2")

    # Logotip qo'shish (agar mavjud bo'lsa)
    try:
        from openpyxl.drawing.image import Image
        import os
        logo_path = "static/logo.png" # Logotip shu joyda bo'lishi kerak
        if os.path.exists(logo_path):
            img = Image(logo_path)
            # O'lchamini proporsional va o'rtacha qilish (original: 1024x173)
            # Factor = ~0.33
            img.width = 338
            img.height = 57
            
            # Rasmni biroz markazlashtiribroq qo'yish uchun katak o'lchamlari
            ws.row_dimensions[1].height = 30
            ws.row_dimensions[2].height = 30
            ws.add_image(img, 'A1')
    except Exception as e:
        print(f"Logotip qo'shishda xatolik: {e}")

    # Sarlavha (3-qatorda)
    total_cols = 3 + len(date_list)
    ws.merge_cells(start_row=3, start_column=1, end_row=3, end_column=total_cols)
    title_cell = ws.cell(row=3, column=1, value=f"Davomat hisoboti — {start_date} dan {end_date} gacha")
    title_cell.font = title_font
    title_cell.alignment = center_align

    # Header qatori (5-qatorda)
    row = 5
    # Asosiy ustunlar
    headers = ["#", "Xodim ID", "To'liq ism"]
    for i, h in enumerate(headers, 1):
        c = ws.cell(row=row, column=i, value=h)
        c.font = bold_font
        c.fill = header_fill
        c.alignment = center_align
        c.border = thin_border

    # Sanalar ustunlari
    for i, d in enumerate(date_list, 4):
        # Sana va hafta kuni nomi (masalan: 01.05\nDu)
        days_uz = ["Du", "Se", "Ch", "Pa", "Ju", "Sh", "Ya"]
        day_name = days_uz[d.weekday()]
        day_str = f"{d.strftime('%d.%m')}\n{day_name}"
        
        c = ws.cell(row=row, column=i, value=day_str)
        c.font = bold_font
        c.alignment = center_align
        c.border = thin_border
        
        # Shanba va Yakshanba - orange
        if d.weekday() >= 5:
            c.fill = weekend_fill
        else:
            c.fill = header_fill

    row += 1

    # Ma'lumotlar qatori
    user_idx = 1
    # Sort users by full name
    sorted_users = sorted(users.items(), key=lambda item: item[1].get("full_name", item[0]))
    
    for wid, uinfo in sorted_users:
        fname = uinfo.get("full_name", wid)
        
        c1 = ws.cell(row=row, column=1, value=user_idx)
        c2 = ws.cell(row=row, column=2, value=wid)
        c3 = ws.cell(row=row, column=3, value=fname)
        
        for c in (c1, c2, c3):
            c.border = thin_border
            c.alignment = center_align if c != c3 else left_align
            
        # Har bir sana uchun yozuvlarni to'ldiramiz
        for i, d in enumerate(date_list, 4):
            d_str = d.isoformat()
            c = ws.cell(row=row, column=i)
            c.border = thin_border
            c.alignment = center_align
            
            if d.weekday() >= 5:
                c.fill = weekend_fill
                
            rec = user_records.get(wid, {}).get(d_str)
            if rec:
                cin_raw = rec.get("check_in_time")
                cout_raw = rec.get("check_out_time")
                cin_disp = cin_raw.split("T")[1][:5] if cin_raw else "-"
                cout_disp = cout_raw.split("T")[1][:5] if cout_raw else "-"
                wh = _format_working_hours(cin_raw, cout_raw)
                
                # Agar ishlagan soati "-" bo'lsa faqat kelish-ketish
                if wh == "-":
                    text = f"K: {cin_disp}\nC: {cout_disp}"
                else:
                    text = f"K: {cin_disp}\nC: {cout_disp}\n🕒 {wh}"
                
                c.value = text
            else:
                c.value = "" # Kelmagan

        row += 1
        user_idx += 1

    # Ustunlar va qatorlar o'lchamlarini moslashtirish
    ws.column_dimensions['B'].width = 15
    ws.column_dimensions['C'].width = 30
    for i in range(4, total_cols + 1):
        col_letter = ws.cell(row=5, column=i).column_letter
        ws.column_dimensions[col_letter].width = 12

    ws.freeze_panes = "D6" # Scrolling qulay bo'lishi uchun

    buf = BytesIO()
    wb.save(buf)

    suffix = f"{start_date}_{end_date}"
    filename = f"davomat_hisoboti_{suffix}.xlsx"

    return Response(
        content=buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


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


# ========== SETTINGS, AUDIT, TELEGRAM ==========






@router.get("/admin/audit")
async def admin_audit_page(request: Request):
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return RedirectResponse(url="/login?next=/admin/audit", status_code=303)
    entries = audit_log.get_recent(300)
    return templates.TemplateResponse(request=request, name="admin_audit.html", context=
        {"request": request, "user": user, "entries": entries},
    )


@router.get("/api/settings/schedule")
async def api_get_schedule(request: Request):
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    return JSONResponse(load_schedule())


@router.put("/api/settings/schedule")
async def api_put_schedule(request: Request, body: dict = Body(...)):
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    ws = (body.get("work_start") or "09:00").strip()
    we = (body.get("work_end") or "18:00").strip()
    save_schedule(ws, we)
    return JSONResponse({"success": True, **load_schedule()})


@router.get("/api/settings/holidays")
async def api_get_holidays(request: Request):
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    return JSONResponse(holidays_db.list_all())


@router.post("/api/settings/holidays")
async def api_post_holiday(request: Request, body: dict = Body(...)):
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    d = (body.get("date") or "").strip()
    if not d:
        return JSONResponse({"error": "date required"}, status_code=400)
    label = (body.get("label") or "").strip()
    holidays_db.add(d, label)
    return JSONResponse({"success": True, "holidays": holidays_db.list_all()})


@router.delete("/api/settings/holidays/{date_str}")
async def api_delete_holiday(request: Request, date_str: str):
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    ok = holidays_db.remove(date_str)
    if not ok:
        return JSONResponse({"success": False}, status_code=404)
    return JSONResponse({"success": True, "holidays": holidays_db.list_all()})


@router.get("/api/audit-log")
async def api_audit_log(request: Request, limit: int = 200):
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    return JSONResponse(audit_log.get_recent(min(limit, 500)))


@router.post("/api/telegram/test")
async def api_telegram_test(request: Request):
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    if not (settings.TELEGRAM_BOT_TOKEN or "").strip():
        return JSONResponse(
            {"ok": False, "error": "TELEGRAM_BOT_TOKEN .env da yo'q"},
            status_code=400,
        )
    if not get_effective_chat_id():
        return JSONResponse(
            {
                "ok": False,
                "error": "Chat ID yo'q. Botga /start yuboring va «Chat ID ni topish» yoki .env da TELEGRAM_CHAT_ID.",
            },
            status_code=400,
        )
    r = send_telegram_message_result(
        "✅ Test: Davomat tizimi — Telegram ulanishi ishlayapti."
    )
    status = 200 if r.get("ok") else 502
    return JSONResponse(r, status_code=status)


@router.post("/api/telegram/send-summary")
async def api_telegram_send_summary(request: Request):
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    if not (settings.TELEGRAM_BOT_TOKEN or "").strip() or not get_effective_chat_id():
        return JSONResponse(
            {
                "ok": False,
                "error": "Token yoki Chat ID to'liq emas (.env yoki Chat ID saqlash).",
            },
            status_code=400,
        )
    text = build_daily_summary_text()
    r = send_telegram_message_result(text)
    status = 200 if r.get("ok") else 502
    return JSONResponse(r, status_code=status)


@router.get("/api/telegram/discover-chats")
async def api_telegram_discover_chats(request: Request):
    """getUpdates orqali chat_id ro'yxati (avval botga xabar yuboring)."""
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    if not (settings.TELEGRAM_BOT_TOKEN or "").strip():
        return JSONResponse({"ok": False, "error": "TELEGRAM_BOT_TOKEN yo'q"}, status_code=400)
    data = fetch_chat_ids_from_updates(80)
    return JSONResponse({"ok": True, **data})


@router.post("/api/settings/telegram-chat")
async def api_save_telegram_chat(request: Request, body: dict = Body(...)):
    """Chat ID ni data/telegram_chat_id.txt ga saqlash (.env dan keyin ustunlik)."""
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    cid = str(body.get("chat_id", "")).strip()
    if not cid:
        return JSONResponse({"ok": False, "error": "chat_id kerak"}, status_code=400)
    save_chat_id_to_file(cid)
    return JSONResponse({"ok": True, "chat_id_saved": True})


@router.get("/api/telegram/managers")
async def api_get_managers(request: Request):
    """Qo'shimcha admin chat ID lar ro'yxatini qaytaradi."""
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    return JSONResponse({"ok": True, "managers": get_extra_chat_ids()})


@router.post("/api/telegram/managers")
async def api_add_manager(request: Request, body: dict = Body(...)):
    """Yangi admin chat ID qo'shadi. Body: {chat_id, label}"""
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    cid = str(body.get("chat_id", "")).strip()
    label = str(body.get("label", "")).strip()
    if not cid:
        return JSONResponse({"ok": False, "error": "chat_id kerak"}, status_code=400)
    entries = get_extra_chat_ids()
    # Dublikat tekshiruvi
    if any(str(e.get("chat_id")) == cid for e in entries):
        return JSONResponse({"ok": False, "error": "Bu chat ID allaqachon qo'shilgan"}, status_code=409)
    entries.append({"chat_id": cid, "label": label or cid})
    save_extra_chat_ids(entries)
    return JSONResponse({"ok": True, "managers": entries})


@router.delete("/api/telegram/managers/{chat_id}")
async def api_remove_manager(request: Request, chat_id: str):
    """Admin chat ID ni o'chiradi."""
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    entries = get_extra_chat_ids()
    new_entries = [e for e in entries if str(e.get("chat_id")) != chat_id]
    if len(new_entries) == len(entries):
        return JSONResponse({"ok": False, "error": "Topilmadi"}, status_code=404)
    save_extra_chat_ids(new_entries)
    return JSONResponse({"ok": True, "managers": new_entries})


@router.post("/api/telegram/send-summary-chart")
async def api_telegram_send_summary_chart(request: Request):
    """Grafik bilan kunlik xulosani barcha chatlarga yuboradi."""
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    if not is_telegram_ready():
        return JSONResponse(
            {"ok": False, "error": "Token yoki Chat ID to'liq emas."},
            status_code=400,
        )
    result = await asyncio.to_thread(send_daily_summary_with_chart)
    status = 200 if result.get("ok") else 502
    return JSONResponse(result, status_code=status)


# ========== DEPARTMENTS ENDPOINTS ==========

@router.get("/admin/departments")
async def admin_departments_page(request: Request):
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return RedirectResponse(url="/login?next=/admin/departments", status_code=303)
    
    return templates.TemplateResponse(request=request, name="admin_departments.html", context={
        "request": request,
        "user": user,
        "departments": departments_db.get_all()
    })

@router.post("/api/departments")
async def api_create_department(request: Request, body: dict = Body(...)):
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Name required"}, status_code=400)
    
    new_dept = departments_db.create(name)
    return JSONResponse({"success": True, "department": new_dept})

@router.put("/api/departments/{dept_id}")
async def api_update_department(request: Request, dept_id: str, body: dict = Body(...)):
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Name required"}, status_code=400)
    
    success = departments_db.update(dept_id, name)
    if success:
        return JSONResponse({"success": True})
    return JSONResponse({"error": "Not found"}, status_code=404)

@router.delete("/api/departments/{dept_id}")
async def api_delete_department(request: Request, dept_id: str):
    user = get_current_admin(request)
    if not user or user.get("role") != "admin":
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    
    success = departments_db.delete(dept_id)
    if success:
        return JSONResponse({"success": True})
    return JSONResponse({"error": "Not found"}, status_code=404)

