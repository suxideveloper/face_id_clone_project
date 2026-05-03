import html
import io
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, date

from app.core.config import settings


# ── Extra admins (multi-recipient) ──────────────────────────────────────────

def _extra_chats_file() -> str:
    return os.path.join(settings.DATA_DIR, "telegram_extra_chats.json")


def get_extra_chat_ids() -> list:
    """Qo'shimcha admin chat ID lar ro'yxatini qaytaradi.
    Har bir element: {chat_id: str, label: str, username: str}"""
    path = _extra_chats_file()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return []


def save_extra_chat_ids(entries: list) -> None:
    """Qo'shimcha admin chat ID larini saqlaydi."""
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    with open(_extra_chats_file(), "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _get_chat_info(chat_id: str) -> dict:
    """Telegram getChat API orqali foydalanuvchi ma'lumotlarini oladi.
    Qaytaradi: {username, first_name, ...} yoki bo'sh dict."""
    result = _tg_api("getChat", {"chat_id": chat_id})
    if result.get("ok"):
        return result.get("result", {})
    return {}


def broadcast_to_all(text: str, parse_mode=None) -> dict:
    """Asosiy chat + barcha qo'shimcha adminlarga xabar yuboradi.
    Natija: {ok: bool, sent: int, errors: list}"""
    main_result = send_telegram_message_result(text, parse_mode=parse_mode)
    errors = []
    sent = 1 if main_result.get("ok") else 0
    if not main_result.get("ok"):
        errors.append(f"main: {main_result.get('error', '?')}")

    for entry in get_extra_chat_ids():
        cid = str(entry.get("chat_id", "")).strip()
        if not cid:
            continue
        r = send_telegram_to_chat(cid, text, parse_mode=parse_mode)
        if r.get("ok"):
            sent += 1
        else:
            errors.append(f"{cid}: {r.get('error', '?')}")

    return {"ok": sent > 0, "sent": sent, "errors": errors}


def broadcast_to_main_only(text: str, parse_mode=None) -> dict:
    """Faqat asosiy admin chatga xabar yuboradi (real-time davomat uchun).
    Qo'shimcha adminlar faqat kunlik hisobotni oladi."""
    main_result = send_telegram_message_result(text, parse_mode=parse_mode)
    return main_result


def _chat_id_file() -> str:
    return os.path.join(settings.DATA_DIR, "telegram_chat_id.txt")


def get_effective_chat_id() -> str:
    """Avval .env (TELEGRAM_CHAT_ID), bo'sh bo'lsa data/telegram_chat_id.txt."""
    e = (settings.TELEGRAM_CHAT_ID or "").strip()
    if e:
        return e
    path = _chat_id_file()
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            pass
    return ""


def save_chat_id_to_file(chat_id: str) -> None:
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    with open(_chat_id_file(), "w", encoding="utf-8") as f:
        f.write(str(chat_id).strip())


def is_telegram_ready() -> bool:
    return bool((settings.TELEGRAM_BOT_TOKEN or "").strip() and get_effective_chat_id())


def _tg_api(method: str, params=None) -> dict:
    token = (settings.TELEGRAM_BOT_TOKEN or "").strip()
    if not token:
        return {"ok": False, "description": "TELEGRAM_BOT_TOKEN yo'q"}
    q = ""
    if params:
        from urllib.parse import urlencode

        q = "?" + urlencode(params, doseq=True)
    url = f"https://api.telegram.org/bot{token}/{method}{q}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        return {"ok": False, "description": str(e)}


def _tg_post(method: str, payload=None) -> dict:
    token = (settings.TELEGRAM_BOT_TOKEN or "").strip()
    if not token:
        return {"ok": False, "description": "TELEGRAM_BOT_TOKEN yo'q"}
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        return {"ok": False, "description": str(e)}


def _offset_file() -> str:
    return os.path.join(settings.DATA_DIR, "telegram_update_offset.txt")


def _read_next_offset() -> int:
    path = _offset_file()
    if not os.path.isfile(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip() or "0")
    except (OSError, ValueError):
        return 0


def _write_next_offset(offset: int) -> None:
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    with open(_offset_file(), "w", encoding="utf-8") as f:
        f.write(str(offset))


def ensure_telegram_long_poll_mode() -> None:
    """Webhook o'rnatilgan bo'lsa getUpdates ishlamaydi — o'chiramiz."""
    token = (settings.TELEGRAM_BOT_TOKEN or "").strip()
    if not token:
        return
    info = _tg_api("getWebhookInfo")
    if not info.get("ok"):
        return
    wh = (info.get("result") or {}).get("url") or ""
    if wh:
        print(f"Telegram: webhook ({wh}) olib tashlanmoqda — long polling uchun.")
        r = _tg_post("deleteWebhook", {"drop_pending_updates": False})
        if r.get("ok"):
            print("Telegram: deleteWebhook OK.")
        else:
            print(f"Telegram: deleteWebhook xato: {r}")


def send_telegram_to_chat(chat_id, text: str, parse_mode=None, reply_markup=None) -> dict:
    """Muayyan chatga xabar (get_effective_chat_id ishlatilmaydi)."""
    token = (settings.TELEGRAM_BOT_TOKEN or "").strip()
    if not token:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN yo'q"}
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if reply_markup:
        payload["reply_markup"] = reply_markup
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            if parsed.get("ok"):
                return {"ok": True}
            return {"ok": False, "error": parsed.get("description", str(parsed))}
    except Exception as e:
        return {"ok": False, "error": str(e)}


bot_admin_states = {}

def _answer_callback_query(callback_query_id: str, text: str = "") -> dict:
    """Telegram callback_query ga javob beradi (loading spinner o'chirish uchun)."""
    params = {"callback_query_id": callback_query_id}
    if text:
        params["text"] = text
    return _tg_api("answerCallbackQuery", params)


def process_telegram_updates_long_poll() -> None:
    """
    Bir marta getUpdates (timeout=5) kutadi va yangilanishlarni qayta ishlaydi.
    Main admin uchun interaktiv tugmalar (Admin qo'shish, hisobot).
    Qo'shimcha adminlar faqat kunlik hisobotni ko'ra oladi.
    """
    token = (settings.TELEGRAM_BOT_TOKEN or "").strip()
    if not token:
        return
    next_offset = _read_next_offset()
    params = {
        "timeout": "5",
        "limit": "100",
    }
    if next_offset > 0:
        params["offset"] = str(next_offset)

    data = _tg_api("getUpdates", params)
    if not data.get("ok"):
        print(f"Telegram getUpdates: {data.get('description', data)}")
        return

    updates = data.get("result") or []
    if not updates:
        return

    main_admin_id = get_effective_chat_id()
    extra_admin_ids = [str(e.get("chat_id", "")).strip() for e in get_extra_chat_ids()]

    for u in updates:
        # ── Callback query (inline button bosilganda) ──
        cb = u.get("callback_query")
        if cb:
            cb_data = (cb.get("data") or "").strip()
            cb_from = cb.get("from") or {}
            cb_cid = str(cb_from.get("id", ""))
            cb_id = cb.get("id", "")

            # Faqat asosiy admin o'chira oladi
            if cb_cid == main_admin_id and cb_data.startswith("rm_admin:"):
                rm_chat_id = cb_data.split(":", 1)[1]
                entries = get_extra_chat_ids()
                removed_entry = None
                new_entries = []
                for e in entries:
                    if str(e.get("chat_id")) == rm_chat_id:
                        removed_entry = e
                    else:
                        new_entries.append(e)
                if removed_entry:
                    save_extra_chat_ids(new_entries)
                    uname = removed_entry.get("username", removed_entry.get("label", rm_chat_id))
                    _answer_callback_query(cb_id, f"✅ {uname} o'chirildi")
                    # Ro'yxatni yangilash uchun yangi xabar yuborish
                    _send_admin_list(cb_cid)
                else:
                    _answer_callback_query(cb_id, "⚠️ Topilmadi")
            else:
                _answer_callback_query(cb_id)
            continue

        msg = u.get("message") or u.get("edited_message") or {}
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        chat = msg.get("chat") or {}
        cid = chat.get("id")
        if cid is None:
            continue
            
        cid_str = str(cid)
        is_main_admin = (cid_str == main_admin_id)
        is_extra_admin = (cid_str in extra_admin_ids)

        if text.lower().startswith("/start"):
            name = (chat.get("first_name") or chat.get("title") or "").strip()
            greet = f"Salom{name and (', ' + name) or ''}! 👋\n\n"
            body = (
                "Bu — <b>AIRI davomat</b> xabarnomasi boti.\n"
                f"Sizning <b>Chat ID</b>:\n<code>{cid}</code>\n\n"
            )
            keyboard = None
            if is_main_admin:
                body += "Siz asosiy adminsiz. Boshqaruv tugmalari orqali botni sozlang:"
                keyboard = {
                    "keyboard": [
                        [{"text": "➕ Admin qo'shish"}, {"text": "👥 Adminlar"}],
                        [{"text": "📊 Kunlik hisobot yuborish"}]
                    ],
                    "resize_keyboard": True
                }
            elif is_extra_admin:
                body += "Siz admin sifatida qo'shilgansiz. Kunlik hisobotni ko'rishingiz mumkin."
                keyboard = {
                    "keyboard": [
                        [{"text": "📊 Kunlik hisobot yuborish"}]
                    ],
                    "resize_keyboard": True
                }
            else:
                body += "Admin panel → <b>Sozlamalar</b> da shu ID ni «Chat ID ni saqlash» orqali qo'shing."
                
            send_telegram_to_chat(cid, greet + body, parse_mode="HTML", reply_markup=keyboard)
            bot_admin_states.pop(cid_str, None)
            continue
            
        # Asosiy admin buyruqlari
        if is_main_admin:
            if text == "➕ Admin qo'shish":
                bot_admin_states[cid_str] = "WAITING_ADMIN_ID"
                send_telegram_to_chat(cid, "Yangi adminning <b>Chat ID</b> sini yuboring:\n(U avval botga /start yuborgan bo'lishi kerak)", parse_mode="HTML")
                continue
                
            elif text == "👥 Adminlar":
                _send_admin_list(cid_str)
                bot_admin_states.pop(cid_str, None)
                continue
                
            elif text == "📊 Kunlik hisobot yuborish":
                send_telegram_to_chat(cid, "Grafik yaratilmoqda...")
                bot_admin_states.pop(cid_str, None)
                try:
                    photo_bytes = build_daily_summary_chart()
                    caption = build_daily_summary_text()
                    _send_photo_to_chat(cid_str, photo_bytes, caption[:1020])
                except Exception as e:
                    send_telegram_to_chat(cid, f"Xato: {e}")
                continue
                
            elif bot_admin_states.get(cid_str) == "WAITING_ADMIN_ID":
                new_admin_id = text.strip()
                entries = get_extra_chat_ids()
                if any(str(e.get("chat_id")) == new_admin_id for e in entries):
                    send_telegram_to_chat(cid, "⚠️ Bu Chat ID allaqachon qo'shilgan!")
                else:
                    # Telegram getChat orqali username va ismni olish
                    chat_info = _get_chat_info(new_admin_id)
                    username = chat_info.get("username", "")
                    first_name = chat_info.get("first_name", "")
                    last_name = chat_info.get("last_name", "")
                    full_name = f"{first_name} {last_name}".strip() or new_admin_id
                    
                    entry = {
                        "chat_id": new_admin_id,
                        "label": full_name,
                        "username": username
                    }
                    entries.append(entry)
                    save_extra_chat_ids(entries)
                    
                    display = f"@{username}" if username else full_name
                    send_telegram_to_chat(
                        cid,
                        f"✅ Admin muvaffaqiyatli qo'shildi:\n"
                        f"👤 {html.escape(display)}\n"
                        f"🆔 <code>{new_admin_id}</code>\n"
                        f"U endi kunlik hisobotni oladi.",
                        parse_mode="HTML"
                    )
                bot_admin_states.pop(cid_str, None)
                continue

        # Qo'shimcha admin buyruqlari (faqat kunlik hisobot)
        if is_extra_admin:
            if text == "📊 Kunlik hisobot yuborish":
                send_telegram_to_chat(cid, "Grafik yaratilmoqda...")
                try:
                    photo_bytes = build_daily_summary_chart()
                    caption = build_daily_summary_text()
                    _send_photo_to_chat(cid_str, photo_bytes, caption[:1020])
                except Exception as e:
                    send_telegram_to_chat(cid, f"Xato: {e}")
                continue

    max_id = max(u["update_id"] for u in updates) + 1
    _write_next_offset(max_id)


def _send_admin_list(chat_id: str) -> None:
    """Adminlar ro'yxatini inline o'chirish tugmalari bilan yuboradi."""
    mgrs = get_extra_chat_ids()
    if not mgrs:
        send_telegram_to_chat(chat_id, "Hozircha qo'shimcha adminlar yo'q.")
        return
    
    lines = ["<b>Qo'shimcha adminlar:</b>\n"]
    inline_buttons = []
    for m in mgrs:
        username = m.get("username", "")
        label = m.get("label", m.get("chat_id"))
        display = f"@{username}" if username else label
        lines.append(f"👤 {html.escape(display)}")
        
        # Inline tugma - o'chirish
        btn_text = f"❌ {display}"
        inline_buttons.append(
            [{"text": btn_text, "callback_data": f"rm_admin:{m.get('chat_id')}"}]
        )
    
    reply_markup = {"inline_keyboard": inline_buttons}
    send_telegram_to_chat(chat_id, "\n".join(lines), parse_mode="HTML", reply_markup=reply_markup)


def fetch_chat_ids_from_updates(limit: int = 50) -> dict:
    """
    getUpdates dan oxirgi xabarlardagi chat_id larni chiqaradi.
    Avval botga (yoki guruhga) biror xabar yuboring yoki /start bosing.
    """
    data = _tg_api("getUpdates", {"limit": str(limit)})
    if not data.get("ok"):
        return {
            "chats": [],
            "error": data.get("description", "getUpdates xato"),
        }
    seen = {}
    for u in data.get("result", []):
        msg = u.get("message") or u.get("edited_message") or {}
        ch = msg.get("chat") or {}
        cid = ch.get("id")
        if cid is None:
            continue
        key = str(cid)
        if key not in seen:
            seen[key] = {
                "chat_id": cid,
                "type": ch.get("type", ""),
                "title": ch.get("title") or ch.get("username") or ch.get("first_name") or "",
            }
    return {"chats": list(seen.values()), "error": None}


def notify_attendance_event(event_type: str, full_name: str, worker_id: str, record: dict) -> None:
    """
    Yuz tanilganda davomat yozilgandan keyin Telegramga qisqa xabar.
    Faqat asosiy admin chatga yuboriladi (qo'shimcha adminlar faqat kunlik hisobot oladi).
    """
    if not settings.TELEGRAM_NOTIFY_ATTENDANCE:
        return
    if not is_telegram_ready():
        return
    fn = html.escape(str(full_name or worker_id))
    wid = html.escape(str(worker_id))
    try:
        if event_type == "check_in":
            cin = record.get("check_in_time")
            t = ""
            if cin:
                t = datetime.fromisoformat(cin).strftime("%H:%M:%S")
            text = (
                f"🟢 <b>Kelish</b>\n"
                f"👤 {fn} <code>{wid}</code>\n"
                f"⏰ {t}"
            )
            broadcast_to_main_only(text, parse_mode="HTML")
        elif event_type == "check_out":
            cin_s = record.get("check_in_time")
            cout_s = record.get("check_out_time")
            wt = "—"
            cin_disp = ""
            cout_disp = ""
            if cin_s:
                cin_dt = datetime.fromisoformat(cin_s)
                cin_disp = cin_dt.strftime("%H:%M:%S")
            if cout_s:
                cout_dt = datetime.fromisoformat(cout_s)
                cout_disp = cout_dt.strftime("%H:%M:%S")
            if cin_s and cout_s:
                dt = datetime.fromisoformat(cout_s) - datetime.fromisoformat(cin_s)
                sec = max(0, int(dt.total_seconds()))
                h, r = divmod(sec, 3600)
                m, _ = divmod(r, 60)
                wt = f"{h}h {m}m"
            text = (
                f"🔵 <b>Ketish</b>\n"
                f"👤 {fn}\n"
                f"⏰ Ketgan: {cout_disp}\n"
                f"Kelgan: {cin_disp}\n"
                f"📊 Ishlangan: {wt}"
            )
            broadcast_to_main_only(text, parse_mode="HTML")
    except Exception as e:
        print(f"Telegram attendance notify: {e}")


def send_telegram_message(text: str, parse_mode=None) -> bool:
    r = send_telegram_message_result(text, parse_mode=parse_mode)
    return r.get("ok", False)


def send_telegram_message_result(text: str, parse_mode=None) -> dict:
    """Natija: {ok: bool, error?: str} — xatolarni tushuntirish uchun."""
    token = (settings.TELEGRAM_BOT_TOKEN or "").strip()
    chat_id = get_effective_chat_id()
    if not token:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN .env da yo'q yoki server qayta ishga tushirilmagan."}
    if not chat_id:
        return {
            "ok": False,
            "error": "Chat ID yo'q. @airi_faceid_bot ga /start yuboring, keyin «Chat ID ni topish» yoki .env ga TELEGRAM_CHAT_ID yozing.",
        }
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            if parsed.get("ok"):
                return {"ok": True}
            desc = parsed.get("description", str(parsed))
            print(f"Telegram sendMessage: {desc}")
            return {"ok": False, "error": desc}
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
            parsed = json.loads(err_body)
            desc = parsed.get("description", err_body)
        except Exception:
            desc = str(e)
        print(f"Telegram HTTP error: {desc}")
        return {"ok": False, "error": desc}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        print(f"Telegram send error: {e}")
        return {"ok": False, "error": str(e)}


def build_daily_summary_text() -> str:
    from app.services.attendance_db import attendance_db
    from app.services.user_db import user_db
    from app.services.holidays_db import holidays_db
    from app.services.schedule_settings import load_schedule, classify_attendance_status

    today = date.today()
    today_iso = today.isoformat()
    hol = holidays_db.get_label(today_iso)
    if hol:
        header = f"📅 {today_iso} — {hol}\n(Ish kuni emas; kelmaganlar hisoblanmaydi.)\n\n"
    else:
        header = f"📅 Kunlik xulosa — {today_iso}\n\n"

    users = user_db.get_all_users()
    records = attendance_db.get_records_by_date(today_iso)
    sched = load_schedule()

    lines = [header]
    lines.append(f"⏰ Ish vaqti: {sched.get('work_start')} – {sched.get('work_end')}\n")
    lines.append(f"👥 Ro'yxatdagilar: {len(users)} | 📝 Bugungi yozuvlar: {len(records)}\n")

    present_names = []
    late_n = 0
    for worker_id, rec in records.items():
        info = user_db.get_user(worker_id) or {}
        name = info.get("full_name", worker_id)
        cin_s = rec.get("check_in_time")
        cout_s = rec.get("check_out_time")
        cin_dt = datetime.fromisoformat(cin_s) if cin_s else None
        cout_dt = datetime.fromisoformat(cout_s) if cout_s else None
        st = classify_attendance_status(cin_dt, cout_dt, today)
        if st.get("late"):
            late_n += 1
        cin_disp = cin_dt.strftime("%H:%M") if cin_dt else "—"
        present_names.append(f"  • {name}: {cin_disp} ({st.get('label', '')})")

    if present_names:
        lines.append("Kelganlar:\n" + "\n".join(sorted(present_names)) + "\n")
    else:
        lines.append("Hozircha kelganlar yo'q.\n")

    if not hol:
        absent = [uid for uid in users if uid not in records]
        lines.append(f"\n🔴 Kelmaganlar: {len(absent)}")
        if absent:
            for uid in sorted(absent)[:40]:
                info = user_db.get_user(uid) or {}
                lines.append(f"  • {info.get('full_name', uid)}")
            if len(absent) > 40:
                lines.append(f"  … va yana {len(absent) - 40} kishi")

    lines.append(f"\n🟠 Kechikkanlar (taxminan): {late_n}")

    return "\n".join(lines)


# ── Grafik (Pillow) ─────────────────────────────────────────────────────────

def build_daily_summary_chart() -> bytes:
    """Bugungi davomat ma'lumotlari asosida PNG rasm (bar chart) chizadi.
    Pillow ishlatiladi — matplotlib talab qilinmaydi.
    Qaytaradi: PNG bytes."""
    from PIL import Image, ImageDraw, ImageFont
    from PIL import Image as PILImage
    from app.services.attendance_db import attendance_db
    from app.services.user_db import user_db
    from app.services.holidays_db import holidays_db
    from app.services.schedule_settings import load_schedule, classify_attendance_status

    today = date.today()
    today_iso = today.isoformat()
    users = user_db.get_all_users()
    records = attendance_db.get_records_by_date(today_iso)
    sched = load_schedule()

    total = len(users)
    present_ids = set(records.keys())
    late_n = 0
    on_time_n = 0
    for wid, rec in records.items():
        cin_s = rec.get("check_in_time")
        cout_s = rec.get("check_out_time")
        cin_dt = datetime.fromisoformat(cin_s) if cin_s else None
        cout_dt = datetime.fromisoformat(cout_s) if cout_s else None
        st = classify_attendance_status(cin_dt, cout_dt, today)
        if st.get("late"):
            late_n += 1
        else:
            on_time_n += 1

    absent_n = max(0, total - len(present_ids))

    # ── High-Res Canvas (Anti-aliasing uchun 2x o'lchamda chizamiz) ──
    scale = 2
    W, H = 800 * scale, 480 * scale
    BG = (18, 18, 30)          # dark navy
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Try to load a font; fall back to default
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26 * scale)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18 * scale)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15 * scale)
    except (OSError, IOError):
        font_title = ImageFont.load_default()
        font_label = font_title
        font_small = font_title

    # Title
    hol = holidays_db.get_label(today_iso)
    title = f"Kunlik davomat — {today_iso}"
    if hol:
        title += f"  ({hol})"
    draw.text((W // 2, 40 * scale), title, font=font_title, fill=(240, 240, 255), anchor="mm")

    # Work time subtitle
    sub = f"Ish vaqti: {sched.get('work_start', '—')} – {sched.get('work_end', '—')}   |   Jami: {total} xodim"
    draw.text((W // 2, 80 * scale), sub, font=font_small, fill=(180, 180, 220), anchor="mm")

    # ── Bar chart ──
    categories = [
        ("O'z vaqtida",  on_time_n, (34, 197, 94)),    # Tailwind green-500
        ("Kechikkan",    late_n,    (245, 158, 11)),   # Tailwind amber-500
        ("Kelmagan",     absent_n,  (239, 68, 68)),    # Tailwind red-500
    ]

    bar_area_x0 = 80 * scale
    bar_area_x1 = W - 60 * scale
    bar_area_y0 = 130 * scale
    bar_area_y1 = H - 110 * scale
    bar_area_w  = bar_area_x1 - bar_area_x0
    bar_area_h  = bar_area_y1 - bar_area_y0

    max_val = max(max(c[1] for c in categories), 4) # Kamida 4 bo'lsin, grid chiroyli chiqishi uchun
    n_bars  = len(categories)
    gap     = 60 * scale
    bar_w   = (bar_area_w - gap * (n_bars + 1)) // n_bars

    # Grid lines
    grid_steps = 4
    for i in range(grid_steps + 1):
        gy = bar_area_y1 - int(i / grid_steps * bar_area_h)
        draw.line([(bar_area_x0, gy), (bar_area_x1, gy)], fill=(45, 45, 65), width=1 * scale)
        val_label = str(int(round(i / grid_steps * max_val)))
        draw.text((bar_area_x0 - 15 * scale, gy), val_label, font=font_small, fill=(140, 140, 180), anchor="rm")

    for i, (cat_name, val, color) in enumerate(categories):
        bx = bar_area_x0 + gap * (i + 1) + bar_w * i
        bar_h_px = int(val / max_val * bar_area_h) if max_val > 0 else 0
        # Minimum bar height — radius (8*scale) dan katta bo'lishi shart
        min_bar_h = 18 * scale
        if val > 0 and bar_h_px < min_bar_h:
            bar_h_px = min_bar_h
        by_top = bar_area_y1 - bar_h_px
        # Xavfsizlik: y koordinatalari to'g'ri ekanligiga ishonch hosil qilish
        by_top = min(by_top, bar_area_y1 - 1)
        cx = bx + bar_w // 2
        # bar_w manfiy bo'lmasligi kerak
        actual_bar_w = max(bar_w, 2 * scale)

        if val > 0:
            shadow_color = tuple(max(0, c - 60) for c in color)
            r = min(8 * scale, bar_h_px // 2, actual_bar_w // 2)  # radius bar o'lchamidan oshmasin
            # Shadow
            draw.rounded_rectangle([bx + 6*scale, by_top + 6*scale, bx + actual_bar_w + 6*scale, bar_area_y1], radius=r, fill=shadow_color)
            # Main Bar
            draw.rounded_rectangle([bx, by_top, bx + actual_bar_w, bar_area_y1], radius=r, fill=color)
            # Value on top
            draw.text((cx, by_top - 20 * scale), str(val), font=font_label, fill=(255, 255, 255), anchor="mm")
        else:
            # Draw a tiny flat line for 0
            draw.rounded_rectangle([bx, bar_area_y1 - 4*scale, bx + actual_bar_w, bar_area_y1], radius=2*scale, fill=color)
            draw.text((cx, bar_area_y1 - 20 * scale), "0", font=font_label, fill=(150, 150, 180), anchor="mm")

        # Category label below
        draw.text((cx, bar_area_y1 + 25 * scale), cat_name, font=font_small, fill=(220, 220, 240), anchor="mt")

    # ── Percentage badge ──
    pct = round(len(present_ids) / total * 100) if total > 0 else 0
    pct_color = (34, 197, 94) if pct >= 80 else (245, 158, 11) if pct >= 50 else (239, 68, 68)
    badge_text = f"Davomat: {pct}%"
    
    # Badge background
    badge_w, badge_h = 220 * scale, 40 * scale
    badge_x0 = (W - badge_w) // 2
    badge_y0 = H - 65 * scale
    
    draw.rounded_rectangle([badge_x0, badge_y0, badge_x0 + badge_w, badge_y0 + badge_h], radius=20*scale, fill=(25, 25, 40), outline=pct_color, width=2*scale)
    draw.text((W // 2, badge_y0 + badge_h // 2), badge_text, font=font_label, fill=pct_color, anchor="mm")

    # ── Downscale for anti-aliasing ──
    final_img = img.resize((W // scale, H // scale), resample=PILImage.Resampling.LANCZOS)

    import io
    buf = io.BytesIO()
    final_img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _send_photo_to_chat(chat_id: str, photo_bytes: bytes, caption: str = "") -> dict:
    """Muayyan chatga foto yuboradi (sendPhoto multipart)."""
    token = (settings.TELEGRAM_BOT_TOKEN or "").strip()
    if not token:
        return {"ok": False, "error": "Token yo'q"}
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    import email.generator
    import email.mime.multipart
    import email.mime.base
    import email.mime.text

    boundary = "----TgBoundary1234567890"
    body_parts = []
    body_parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
        f"{chat_id}\r\n"
    )
    if caption:
        body_parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="caption"\r\n\r\n'
            f"{caption}\r\n"
        )
    body_parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="photo"; filename="chart.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    )
    body = "".join(body_parts).encode("utf-8") + photo_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8", errors="replace"))
            return {"ok": result.get("ok", False), "error": result.get("description")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def send_daily_summary_with_chart() -> dict:
    """Barcha chatlarga grafik + matn xulosasini yuboradi.
    Natija: {ok: bool, sent: int, errors: list}"""
    if not is_telegram_ready():
        return {"ok": False, "sent": 0, "errors": ["Telegram sozlanmagan"]}

    try:
        photo_bytes = build_daily_summary_chart()
    except Exception as e:
        print(f"Chart generation error: {e}")
        # Fallback: faqat matn
        text = build_daily_summary_text()
        return broadcast_to_all(text)

    caption = build_daily_summary_text()
    # Telegram caption limit 1024 belgi
    if len(caption) > 1024:
        caption = caption[:1020] + "..."

    all_chats = [get_effective_chat_id()] + [
        str(e.get("chat_id", "")).strip()
        for e in get_extra_chat_ids()
        if str(e.get("chat_id", "")).strip()
    ]

    sent = 0
    errors = []
    for cid in all_chats:
        if not cid:
            continue
        r = _send_photo_to_chat(cid, photo_bytes, caption)
        if r.get("ok"):
            sent += 1
        else:
            errors.append(f"{cid}: {r.get('error', '?')}")

    return {"ok": sent > 0, "sent": sent, "errors": errors}
