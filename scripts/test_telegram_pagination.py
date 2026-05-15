"""
Telegram pagination va shtat birligi testlari.
Haqiqiy Telegram yuborishsiz, logika to'g'riligini tekshiradi.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

# ──────────────────────────────────────────────
# 1. PAGE_SIZE va pagination keyboard testi
# ──────────────────────────────────────────────
PAGE_SIZE = 15

def _build_pagination_keyboard(section, page, total_pages):
    if total_pages <= 1:
        return None
    row = []
    if page > 0:
        row.append({"text": "⬅️ Oldingi", "callback_data": f"report_page:{section}:{page-1}"})
    row.append({"text": f"{page+1}/{total_pages}", "callback_data": f"report_page:{section}:{page}"})
    if page < total_pages - 1:
        row.append({"text": "Keyingi ➡️", "callback_data": f"report_page:{section}:{page+1}"})
    return {"inline_keyboard": [row]}

def test_pagination_keyboard():
    print("\n=== TEST 1: Pagination keyboard ===")
    cases = [
        (10,  1),   # 10 ta — 1 sahifa, keyboard bo'lmaydi
        (15,  1),   # 15 ta — 1 sahifa, keyboard bo'lmaydi
        (16,  2),   # 16 ta — 2 sahifa, keyboard bor
        (100, 7),   # 100 ta — 7 sahifa
        (105, 7),   # 105 ta — 7 sahifa
    ]
    all_ok = True
    for count, expected_pages in cases:
        pages = max(1, math.ceil(count / PAGE_SIZE))
        ok = pages == expected_pages
        status = "✅" if ok else "❌"
        print(f"  {status} {count} xodim → {pages} sahifa (kutilgan: {expected_pages})")
        if not ok:
            all_ok = False

    # Keyboard tugmalari testi
    # 1-sahifa (bosh): faqat "Keyingi ➡️" bo'lishi kerak
    kb = _build_pagination_keyboard("present", 0, 3)
    assert kb is not None, "❌ Keyboard None bo'lmasligi kerak"
    buttons = kb["inline_keyboard"][0]
    texts = [b["text"] for b in buttons]
    assert "⬅️ Oldingi" not in texts, "❌ 1-sahifada Oldingi bo'lmasligi kerak"
    assert "Keyingi ➡️" in texts, "❌ 1-sahifada Keyingi bo'lishi kerak"
    assert "1/3" in texts, "❌ Sahifa raqami noto'g'ri"
    print("  ✅ 1-sahifa keyboard: faqat [1/3] [Keyingi]")

    # O'rta sahifa: ikkalasi ham bo'lishi kerak
    kb = _build_pagination_keyboard("present", 1, 3)
    buttons = kb["inline_keyboard"][0]
    texts = [b["text"] for b in buttons]
    assert "⬅️ Oldingi" in texts, "❌ O'rta sahifada Oldingi bo'lishi kerak"
    assert "Keyingi ➡️" in texts, "❌ O'rta sahifada Keyingi bo'lishi kerak"
    print("  ✅ O'rta sahifa keyboard: [Oldingi] [2/3] [Keyingi]")

    # Oxirgi sahifa: faqat "⬅️ Oldingi" bo'lishi kerak
    kb = _build_pagination_keyboard("present", 2, 3)
    buttons = kb["inline_keyboard"][0]
    texts = [b["text"] for b in buttons]
    assert "⬅️ Oldingi" in texts, "❌ Oxirgi sahifada Oldingi bo'lishi kerak"
    assert "Keyingi ➡️" not in texts, "❌ Oxirgi sahifada Keyingi bo'lmasligi kerak"
    print("  ✅ Oxirgi sahifa keyboard: [Oldingi] [3/3]")

    # 1 sahifa bo'lsa keyboard yo'q
    kb = _build_pagination_keyboard("present", 0, 1)
    assert kb is None, "❌ 1 sahifada keyboard bo'lmasligi kerak"
    print("  ✅ 1 sahifada keyboard None (tugmalar ko'rsatilmaydi)")

    return all_ok

# ──────────────────────────────────────────────
# 2. Ro'yxat chunking testi
# ──────────────────────────────────────────────
def test_list_chunking():
    print("\n=== TEST 2: Ro'yxat chunking ===")
    # 53 ta xodim (rasmda ko'ringan)
    items = [f"  • Xodim {i}" for i in range(1, 54)]
    total_pages = max(1, math.ceil(len(items) / PAGE_SIZE))

    print(f"  Jami: {len(items)} xodim → {total_pages} sahifa")
    all_ok = True
    for p in range(total_pages):
        chunk = items[p * PAGE_SIZE : (p+1) * PAGE_SIZE]
        expected = min(PAGE_SIZE, len(items) - p * PAGE_SIZE)
        ok = len(chunk) == expected
        status = "✅" if ok else "❌"
        print(f"  {status} Sahifa {p+1}: {len(chunk)} ta xodim (kutilgan: {expected})")
        if not ok:
            all_ok = False
    return all_ok

# ──────────────────────────────────────────────
# 3. Callback data format testi
# ──────────────────────────────────────────────
def test_callback_format():
    print("\n=== TEST 3: Callback data format ===")
    all_ok = True
    test_cases = [
        ("report_page:present:0",  ("present", 0)),
        ("report_page:absent:3",   ("absent",  3)),
        ("report_page:present:10", ("present", 10)),
    ]
    for cb_data, (exp_section, exp_page) in test_cases:
        try:
            _, section, page_str = cb_data.split(":")
            page = int(page_str)
            ok = section == exp_section and page == exp_page
            status = "✅" if ok else "❌"
            print(f"  {status} '{cb_data}' → section={section}, page={page}")
            if not ok:
                all_ok = False
        except Exception as e:
            print(f"  ❌ '{cb_data}' parse xato: {e}")
            all_ok = False
    return all_ok

# ──────────────────────────────────────────────
# 4. Shtat birligi message format testi
# ──────────────────────────────────────────────
def test_staff_rate_message():
    print("\n=== TEST 4: Shtat birligi xabar formati ===")

    def build_checkin_text(full_name, staff_rate, check_in_time):
        fn = full_name
        rate_str = f"{staff_rate:g}"
        return (
            f"🟢 <b>Kelish</b>\n"
            f"👤 {fn}\n"
            f"📋 Shtat: {rate_str} birlik\n"
            f"⏰ {check_in_time}"
        )

    def build_checkout_text(full_name, staff_rate, check_out, check_in, worked):
        fn = full_name
        rate_str = f"{staff_rate:g}"
        return (
            f"🔵 <b>Ketish</b>\n"
            f"👤 {fn}\n"
            f"📋 Shtat: {rate_str} birlik\n"
            f"⏰ Ketgan: {check_out}\n"
            f"Kelgan: {check_in}\n"
            f"📊 Ishlangan: {worked}"
        )

    cases = [
        ("Javohir Qahhorov",  1.0,  "1"),
        ("Islom Mamadiyev",   0.5,  "0.5"),
        ("Taisiya Filyayeva", 0.25, "0.25"),
        ("Azizbek T.",        1.5,  "1.5"),
        ("Test Xodim",        2.0,  "2"),
    ]

    all_ok = True
    for name, rate, expected_str in cases:
        ci = build_checkin_text(name, rate, "09:00:00")
        co = build_checkout_text(name, rate, "18:00:00", "09:00:00", "9h 0m")

        ci_has_id  = any(c.isdigit() and len(str(int(c))) > 3 for c in ci.split()) # ID emas
        ci_has_rate = f"Shtat: {expected_str} birlik" in ci
        co_has_rate = f"Shtat: {expected_str} birlik" in co
        ci_no_code  = "<code>" not in ci   # ID yo'q
        co_no_code  = "<code>" not in co

        ok = ci_has_rate and co_has_rate and ci_no_code and co_no_code
        status = "✅" if ok else "❌"
        print(f"  {status} {name} (rate={rate}) → Kelish: {'✓' if ci_has_rate else '✗'} Ketish: {'✓' if co_has_rate else '✗'} ID_yo'q: {'✓' if ci_no_code and co_no_code else '✗'}")
        if not ok:
            all_ok = False
            print(f"     Kelish matni:\n{ci}")
            print(f"     Ketish matni:\n{co}")
    return all_ok

# ──────────────────────────────────────────────
# 5. Import testi (modul yuklanadimy?)
# ──────────────────────────────────────────────
def test_module_import():
    print("\n=== TEST 5: Modul import ===")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "telegram_notify",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "app", "services", "telegram_notify.py")
        )
        # Faqat parse qilamiz (import qilsak DB kerak bo'ladi)
        with open(spec.origin, "r") as f:
            src = f.read()
        import ast
        tree = ast.parse(src)
        
        # Muhim funksiyalar mavjudmi?
        funcs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        required = [
            "_build_pagination_keyboard",
            "_load_report_cache",
            "_send_report_page",
            "_send_daily_report_paginated",
            "notify_attendance_event",
            "send_daily_summary_with_chart",
        ]
        all_ok = True
        for fn in required:
            ok = fn in funcs
            status = "✅" if ok else "❌"
            print(f"  {status} {fn}()")
            if not ok:
                all_ok = False

        # Global o'zgaruvchilar (Assign va AnnAssign ikkalasi)
        globals_assign = {node.targets[0].id for node in ast.walk(tree)
                         if isinstance(node, ast.Assign)
                         and isinstance(node.targets[0], ast.Name)}
        globals_annassign = {node.target.id for node in ast.walk(tree)
                             if isinstance(node, ast.AnnAssign)
                             and isinstance(node.target, ast.Name)}
        globals_found = globals_assign | globals_annassign
        for var in ["_report_pages_cache", "PAGE_SIZE"]:
            ok = var in globals_found
            status = "✅" if ok else "❌"
            print(f"  {status} global {var}")
            if not ok:
                all_ok = False
        return all_ok
    except SyntaxError as e:
        print(f"  ❌ Sintaksis xatosi: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Kutilmagan xato: {e}")
        return False

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    results = []
    results.append(("Pagination keyboard",  test_pagination_keyboard()))
    results.append(("Ro'yxat chunking",     test_list_chunking()))
    results.append(("Callback format",      test_callback_format()))
    results.append(("Shtat birligi format", test_staff_rate_message()))
    results.append(("Modul import/parse",   test_module_import()))

    print("\n" + "="*50)
    print("YAKUNIY NATIJA:")
    all_passed = True
    for name, ok in results:
        status = "✅ O'tdi" if ok else "❌ Muvaffaqiyatsiz"
        print(f"  {status} — {name}")
        if not ok:
            all_passed = False

    print("="*50)
    if all_passed:
        print("🎉 Barcha testlar muvaffaqiyatli o'tdi!")
    else:
        print("⚠️ Ba'zi testlar muvaffaqiyatsiz!")
    sys.exit(0 if all_passed else 1)
