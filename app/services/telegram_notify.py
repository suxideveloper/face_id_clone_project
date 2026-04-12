import html
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, date

from app.core.config import settings


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
        with urllib.request.urlopen(req, timeout=70) as resp:
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


def send_telegram_to_chat(chat_id, text: str, parse_mode=None) -> dict:
    """Muayyan chatga xabar (get_effective_chat_id ishlatilmaydi)."""
    token = (settings.TELEGRAM_BOT_TOKEN or "").strip()
    if not token:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN yo'q"}
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
            return {"ok": False, "error": parsed.get("description", str(parsed))}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def process_telegram_updates_long_poll() -> None:
    """
    Bir marta getUpdates (timeout=50) kutadi va yangilanishlarni qayta ishlaydi.
    /start ga javob + Chat ID ko'rsatish.
    """
    token = (settings.TELEGRAM_BOT_TOKEN or "").strip()
    if not token:
        return
    next_offset = _read_next_offset()
    params = {
        "timeout": "50",
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

    for u in updates:
        msg = u.get("message") or u.get("edited_message") or {}
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        low = text.split()[0].lower() if text else ""
        if not low.startswith("/start"):
            continue
        chat = msg.get("chat") or {}
        cid = chat.get("id")
        if cid is None:
            continue
        name = (chat.get("first_name") or chat.get("title") or "").strip()
        greet = f"Salom{name and (', ' + name) or ''}! 👋\n\n"
        body = (
            "Bu — <b>AIRI davomat</b> xabarnomasi boti.\n"
            f"Sizning <b>Chat ID</b>:\n<code>{cid}</code>\n\n"
            "Admin panel → <b>Sozlamalar</b> da shu ID ni «Chat ID ni saqlash» orqali qo'shing, "
            "keyin test xabar yuboring."
        )
        send_telegram_to_chat(cid, greet + body, parse_mode="HTML")

    max_id = max(u["update_id"] for u in updates) + 1
    _write_next_offset(max_id)


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
    Faqat token + chat ID va TELEGRAM_NOTIFY_ATTENDANCE yoqilgan bo'lsa.
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
            send_telegram_message_result(text, parse_mode="HTML")
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
            send_telegram_message_result(text, parse_mode="HTML")
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
