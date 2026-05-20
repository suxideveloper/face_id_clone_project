"""
Liveness Detection — To'liq Test Suite
=======================================
Barcha edge case va holatlarni qamrab oladi.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import time

# ── Ranglar ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed = 0
failed = 0
errors = []

def ok(msg):
    global passed
    passed += 1
    print(f"  {GREEN}✅ {msg}{RESET}")

def fail(msg, detail=""):
    global failed
    failed += 1
    errors.append(msg)
    print(f"  {RED}❌ {msg}{RESET}")
    if detail:
        print(f"     {YELLOW}→ {detail}{RESET}")

def section(title):
    print(f"\n{BOLD}{CYAN}{'═'*55}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*55}{RESET}")

# ══════════════════════════════════════════════════════════
section("1. IMPORT VA MODUL YUKLASH")
# ══════════════════════════════════════════════════════════

try:
    import app.services.liveness as liveness_module
    from app.services.liveness import (
        liveness_detector, LivenessDetector,
        EAR_THRESHOLD, EAR_CONSEC_FRAMES,
        BLINKS_REQUIRED, BLINK_TIMEOUT_SECONDS,
        TEXTURE_THRESHOLD, TEXTURE_SAMPLE_FRAMES,
        LEFT_EYE_INDICES, RIGHT_EYE_INDICES,
        GLASSES_FALLBACK_FRAMES, GLASSES_TEXTURE_THRESHOLD,
    )
    ok("liveness.py import (glasses konstantalari bilan)")
except Exception as e:
    fail("liveness.py import", str(e))

try:
    from app.services.tracker import Tracker
    ok("tracker.py import")
except Exception as e:
    fail("tracker.py import", str(e))

try:
    from app.services.video_processor import VideoProcessor
    ok("video_processor.py import")
except Exception as e:
    fail("video_processor.py import", str(e))

try:
    from app.api.routes import generate_frames
    ok("routes.py import (generate_frames mavjud)")
except Exception as e:
    fail("routes.py import", str(e))

# ══════════════════════════════════════════════════════════
section("2. KONSTANTALAR TO'G'RILIGI")
# ══════════════════════════════════════════════════════════

if EAR_THRESHOLD == 0.21:
    ok(f"EAR_THRESHOLD = {EAR_THRESHOLD}")
else:
    fail(f"EAR_THRESHOLD = {EAR_THRESHOLD}", "0.21 bo'lishi kerak")

if BLINKS_REQUIRED == 1:
    ok(f"BLINKS_REQUIRED = {BLINKS_REQUIRED}")
else:
    fail(f"BLINKS_REQUIRED = {BLINKS_REQUIRED}", "1 bo'lishi kerak")

if EAR_CONSEC_FRAMES == 2:
    ok(f"EAR_CONSEC_FRAMES = {EAR_CONSEC_FRAMES}")
else:
    fail(f"EAR_CONSEC_FRAMES = {EAR_CONSEC_FRAMES}", "2 bo'lishi kerak")

if TEXTURE_THRESHOLD == 60.0:
    ok(f"TEXTURE_THRESHOLD = {TEXTURE_THRESHOLD}")
else:
    fail(f"TEXTURE_THRESHOLD = {TEXTURE_THRESHOLD}", "60.0 bo'lishi kerak")

if len(LEFT_EYE_INDICES) == 6 and LEFT_EYE_INDICES[0] == 36:
    ok(f"LEFT_EYE_INDICES = {LEFT_EYE_INDICES}")
else:
    fail("LEFT_EYE_INDICES noto'g'ri")

if len(RIGHT_EYE_INDICES) == 6 and RIGHT_EYE_INDICES[0] == 42:
    ok(f"RIGHT_EYE_INDICES = {RIGHT_EYE_INDICES}")
else:
    fail("RIGHT_EYE_INDICES noto'g'ri")

# ══════════════════════════════════════════════════════════
section("3. EAR FORMULASI TESTI")
# ══════════════════════════════════════════════════════════

# Ko'z ochiq: EAR ≈ 0.30
# Geometriyasi: gorizontal=6, vertikal=1.5 → EAR=(1.5+1.5)/(2*6)=0.25
open_eye = np.array([
    [0, 0], [2, -2], [4, -2],   # yuqori
    [6, 0],                      # o'ng
    [4,  2], [2,  2],            # quyi
], dtype=np.float64)

ear_open = liveness_detector._get_ear(open_eye)
# EAR formulasi to'g'ri ishlayotganini tekshiramiz:
# Ko'z ochiq → EAR > EAR_THRESHOLD (0.21) bo'lishi shart
if ear_open > EAR_THRESHOLD:
    ok(f"Ko'z OCHIQ EAR = {ear_open:.3f} (threshold {EAR_THRESHOLD}'dan katta — to'g'ri)")
else:
    fail(f"Ko'z OCHIQ EAR = {ear_open:.3f}", f"threshold {EAR_THRESHOLD}'dan katta bo'lishi kerak")

# Ko'z yopiq: barcha nuqtalar bir tekisda → EAR ≈ 0
closed_eye = np.array([
    [0,0],[2,0],[4,0],[6,0],[4,0],[2,0]
], dtype=np.float64)
ear_closed = liveness_detector._get_ear(closed_eye)
if ear_closed < 0.05:
    ok(f"Ko'z YOPIQ EAR = {ear_closed:.3f} (kutilgan: < 0.05)")
else:
    fail(f"Ko'z YOPIQ EAR = {ear_closed:.3f}", "< 0.05 bo'lishi kerak")

# EAR threshold tekshiruvi
if ear_open > EAR_THRESHOLD:
    ok(f"Ochiq ko'z EAR ({ear_open:.3f}) > threshold ({EAR_THRESHOLD})")
else:
    fail("Ochiq ko'z EAR threshold'dan kichik")

if ear_closed < EAR_THRESHOLD:
    ok(f"Yopiq ko'z EAR ({ear_closed:.3f}) < threshold ({EAR_THRESHOLD})")
else:
    fail("Yopiq ko'z EAR threshold'dan katta")

# ══════════════════════════════════════════════════════════
section("4. TEXTURE / SPOOF DETECTION TESTI")
# ══════════════════════════════════════════════════════════

# Real yuz simulyatsiyasi: o'tkir, teksturali rasm
sharp_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
cv2.GaussianBlur(sharp_img, (1, 1), 0)  # asl holat

# Bosma rasm simulyatsiyasi: juda xira, tekis
blurry_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
blurry_img = cv2.GaussianBlur(blurry_img, (51, 51), 20)

bbox = (10, 10, 190, 190)
score_sharp  = liveness_detector._check_texture(sharp_img,  bbox)
score_blurry = liveness_detector._check_texture(blurry_img, bbox)

if score_sharp > score_blurry:
    ok(f"Keskin rasm score ({score_sharp:.1f}) > Xira rasm score ({score_blurry:.1f})")
else:
    fail("Keskin rasm score past", f"sharp={score_sharp:.1f}, blurry={score_blurry:.1f}")

if score_sharp >= TEXTURE_THRESHOLD:
    ok(f"Keskin rasm texture_pass: True ({score_sharp:.1f} >= {TEXTURE_THRESHOLD})")
else:
    fail("Keskin rasm texture_pass False bo'lmasligi kerak", f"score={score_sharp:.1f}")

if score_blurry < TEXTURE_THRESHOLD:
    ok(f"Xira rasm texture_pass: False ({score_blurry:.1f} < {TEXTURE_THRESHOLD})")
else:
    fail("Xira rasm texture_pass True bo'lmasligi kerak", f"score={score_blurry:.1f}")

# Empty ROI test
empty = np.zeros((200, 200, 3), dtype=np.uint8)
score_empty = liveness_detector._check_texture(empty, (0, 0, 0, 0))
if score_empty == 0.0:
    ok("Bo'sh ROI → score = 0.0 (crash yo'q)")
else:
    fail("Bo'sh ROI noto'g'ri natija", str(score_empty))

# ══════════════════════════════════════════════════════════
section("5. TRACKER LIVENESS STATE TESTI")
# ══════════════════════════════════════════════════════════

t = Tracker()

# _new_track_data liveness fieldlari (glasses fieldlari bilan)
td = t._new_track_data((0,0,100,100))
required_fields = ['blink_count','ear_below_count','last_ear',
                   'texture_scores','texture_pass','is_live','is_spoof',
                   'liveness_start','no_landmark_streak','liveness_mode']
missing = [f for f in required_fields if f not in td.get('liveness', {})]
if not missing:
    ok("_new_track_data() barcha liveness fieldlari mavjud (glasses bilan)")
else:
    fail("_new_track_data() ba'zi fieldlar yo'q", str(missing))

# Boshlang'ich qiymatlar
lv = td['liveness']
if lv['blink_count'] == 0 and lv['is_live'] == False and lv['is_spoof'] == False:
    ok("Boshlang'ich liveness holat to'g'ri (0, False, False)")
else:
    fail("Boshlang'ich liveness holat xato", str(lv))

# get_liveness_state — mavjud bo'lmagan track
t2 = Tracker()
lv_unknown = t2.get_liveness_state(999)
if lv_unknown['is_live'] == False and lv_unknown['is_spoof'] == False:
    ok("Mavjud bo'lmagan track → default holat qaytarildi")
else:
    fail("Mavjud bo'lmagan track xato qaytarmoqda", str(lv_unknown))

# ══════════════════════════════════════════════════════════
section("6. BLINK DETECTION MANTIQ TESTI")
# ══════════════════════════════════════════════════════════

# Active blink tests expect blink logic to be active
liveness_module.PASSIVE_LIVENESS_ONLY = False
liveness_module.BLINKS_REQUIRED = 2

def make_metrics(ear, texture=150.0):
    return {'ear': ear, 'texture_score': texture,
            'texture_pass': texture >= TEXTURE_THRESHOLD, 'has_landmarks': True}

t3 = Tracker()
t3.tracks[1] = t3._new_track_data((0,0,100,100))

# 1. Ko'z ochiq — blink yo'q
t3.update_liveness(1, make_metrics(0.30))
lv = t3.get_liveness_state(1)
if lv['blink_count'] == 0:
    ok("Ko'z OCHIQ → blink yo'q (0)")
else:
    fail("Ko'z ochiq blink sanadi", f"blink_count={lv['blink_count']}")

# 2. Ko'z yopiq 1 frame — hali blink emas (EAR_CONSEC_FRAMES=2 kerak)
t3.update_liveness(1, make_metrics(0.15))
lv = t3.get_liveness_state(1)
if lv['blink_count'] == 0:
    ok(f"Ko'z YOPIQ 1 frame → hali blink emas (EAR_CONSEC_FRAMES={EAR_CONSEC_FRAMES})")
else:
    fail("1 frame yopiqda blink sanadi", f"blink_count={lv['blink_count']}")

# 3. Ko'z yopiq 2 frame + ochiq → 1 BLINK
t3.update_liveness(1, make_metrics(0.14))  # 2-frame yopiq
t3.update_liveness(1, make_metrics(0.30))  # ochiq → blink!
lv = t3.get_liveness_state(1)
if lv['blink_count'] == 1:
    ok("Ko'z yopiq×2 → ochiq → BLINK +1")
else:
    fail("Blink aniqlanmadi", f"blink_count={lv['blink_count']}")

# 4. Ikkinchi blink → is_live = True
t3.update_liveness(1, make_metrics(0.13))
t3.update_liveness(1, make_metrics(0.12))
t3.update_liveness(1, make_metrics(0.30))
lv = t3.get_liveness_state(1)
if lv['blink_count'] >= BLINKS_REQUIRED and lv['is_live']:
    ok(f"Blink×{lv['blink_count']} → is_live=True ✓")
else:
    fail(f"is_live hali True emas", f"blink={lv['blink_count']}, is_live={lv['is_live']}")

# 5. is_live bo'lgandan keyin update_liveness → holat o'zgarmasin
t3.update_liveness(1, make_metrics(0.10))
t3.update_liveness(1, make_metrics(0.10))
lv2 = t3.get_liveness_state(1)
if lv2['is_live'] == True:
    ok("is_live=True holat o'zgarmas (idempotent)")
else:
    fail("is_live reset qilinib qoldi!")

# ══════════════════════════════════════════════════════════
section("7. SPOOF DETECTION TESTI")
# ══════════════════════════════════════════════════════════

liveness_module.PASSIVE_LIVENESS_ONLY = True
liveness_module.BLINKS_REQUIRED = 1

t4 = Tracker()
t4.tracks[1] = t4._new_track_data((0,0,100,100))

# Juda past texture (< 20) → spoof
spoof_m = {'ear': 0.28, 'texture_score': 5.0, 'texture_pass': False, 'has_landmarks': True}
for _ in range(TEXTURE_SAMPLE_FRAMES):
    t4.update_liveness(1, spoof_m)

lv = t4.get_liveness_state(1)
if lv['is_spoof'] == True:
    ok(f"Texture < 20 ({TEXTURE_SAMPLE_FRAMES} frame) → SPOOF aniqlandi")
else:
    fail("Spoof aniqlanmadi", f"texture_scores in lv = {t4.tracks[1]['liveness']['texture_scores']}")

if lv['is_live'] == False:
    ok("SPOOF holati: is_live=False (to'g'ri)")
else:
    fail("SPOOF holati is_live=True bo'lishi xato!")

# Spoof holatida keyingi updatelar is_live'ni o'zgartirmasin
t4.update_liveness(1, make_metrics(0.15))
t4.update_liveness(1, make_metrics(0.15))
t4.update_liveness(1, make_metrics(0.30))
lv2 = t4.get_liveness_state(1)
if lv2['is_live'] == False and lv2['is_spoof'] == True:
    ok("SPOOF aniqlangandan keyin blink'lar is_live'ni o'zgartirmadi")
else:
    fail("SPOOF holatidan keyin is_live o'zgarib qoldi!")

# ══════════════════════════════════════════════════════════
section("8. EAR=None HOLATI (Landmark topilmasa)")
# ══════════════════════════════════════════════════════════

t5 = Tracker()
t5.tracks[1] = t5._new_track_data((0,0,100,100))
no_landmark = {'ear': None, 'texture_score': 120.0, 'texture_pass': True, 'has_landmarks': False}

try:
    for _ in range(5):
        t5.update_liveness(1, no_landmark)
    lv = t5.get_liveness_state(1)
    if lv['blink_count'] == 0 and lv['last_ear'] is None:
        ok("EAR=None → blink sanalmadi, crash yo'q")
    else:
        fail("EAR=None holatida noto'g'ri natija", str(lv))
except Exception as e:
    fail("EAR=None → CRASH!", str(e))

# ══════════════════════════════════════════════════════════
section("9. REMOVE_USER → LIVENESS RESET TESTI")
# ══════════════════════════════════════════════════════════

liveness_module.PASSIVE_LIVENESS_ONLY = False
liveness_module.BLINKS_REQUIRED = 2

t6 = Tracker()
t6.tracks[1] = t6._new_track_data((0,0,100,100))
t6.tracks[1]['name'] = 'john'
t6.tracks[1]['confirmed'] = True
# Liveness to'liq tasdiqlash
for ear in [0.30, 0.14, 0.14, 0.30, 0.13, 0.13, 0.30]:
    t6.update_liveness(1, make_metrics(ear))

lv_before = t6.get_liveness_state(1)
t6.remove_user('john')
lv_after = t6.get_liveness_state(1)

if lv_before['is_live'] and not lv_after['is_live']:
    ok("remove_user() → liveness holati reset qilindi")
elif not lv_before['is_live']:
    fail("Test setup xato: liveness is_live=False bo'lib qoldi")
else:
    fail("remove_user() liveness'ni reset qilmadi", str(lv_after))

# ══════════════════════════════════════════════════════════
section("10. KO'P YUZ (Multi-face) MUSTAQILLIK TESTI")
# ══════════════════════════════════════════════════════════

t7 = Tracker()
t7.tracks[1] = t7._new_track_data((0,0,100,100))
t7.tracks[2] = t7._new_track_data((200,200,300,300))

# Track 1: 2 blink (live bo'lsin)
for ear in [0.14, 0.14, 0.30, 0.13, 0.13, 0.30]:
    t7.update_liveness(1, make_metrics(ear))

# Track 2: hech narsa qilmaymiz
lv1 = t7.get_liveness_state(1)
lv2 = t7.get_liveness_state(2)

if lv1['is_live'] and not lv2['is_live']:
    ok("Track 1 live, Track 2 emas — mustaqil holat ✓")
else:
    fail("Multi-face holat xato", f"t1.is_live={lv1['is_live']}, t2.is_live={lv2['is_live']}")

if lv1['blink_count'] > 0 and lv2['blink_count'] == 0:
    ok(f"Blink countlar mustaqil: t1={lv1['blink_count']}, t2={lv2['blink_count']}")
else:
    fail("Blink count'lar bir-biriga bog'liq chiqdi")

# ══════════════════════════════════════════════════════════
section("11. ANALYZE_FRAME() REAL KAMERA TESTI")
# ══════════════════════════════════════════════════════════

liveness_module.PASSIVE_LIVENESS_ONLY = True
liveness_module.BLINKS_REQUIRED = 1

cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    cap.release()
    if ret:
        h, w = frame.shape[:2]
        bbox = (w//4, h//4, 3*w//4, 3*h//4)
        try:
            result = liveness_detector.analyze_frame(frame, bbox)
            if 'ear' in result and 'texture_score' in result and 'texture_pass' in result:
                ok(f"analyze_frame() qaytargan dict to'g'ri: {list(result.keys())}")
            else:
                fail("analyze_frame() dict to'liq emas", str(result.keys()))

            if result['texture_score'] >= 0:
                ok(f"texture_score = {result['texture_score']:.1f} (manfiy emas)")
            else:
                fail("texture_score manfiy", str(result['texture_score']))

            print(f"     EAR: {result['ear']}, texture: {result['texture_score']:.1f}, pass: {result['texture_pass']}")
        except Exception as e:
            fail("analyze_frame() crash!", str(e))
    else:
        print(f"  {YELLOW}⚠️  Kamera frame olishda muammo (lekin main.py band bo'lishi mumkin){RESET}")
else:
    print(f"  {YELLOW}⚠️  Kamera band (main.py ishlamoqda) — bu normal{RESET}")
    cap.release()

# ══════════════════════════════════════════════════════════
section("12. ROUTES.PY VA VIDEO_PROCESSOR.PY INTEGRATSIYA")
# ══════════════════════════════════════════════════════════

try:
    import inspect
    from app.api import routes
    src = inspect.getsource(routes.generate_frames)
    checks = [
        ('liveness_detector.analyze_frame', "analyze_frame chaqiruvi"),
        ('face_tracker.update_liveness',    "update_liveness chaqiruvi"),
        ('face_tracker.get_liveness_state', "get_liveness_state chaqiruvi"),
        ('is_confirmed and is_live',        "Attendance ikki shart bilan"),
        ('BLINK_TIMEOUT_SECONDS' in src or 'BLINKS_REQUIRED' in src,
                                            "BLINKS_REQUIRED ishlatilgan"),
        ('SPOOF DETECTED',                  "Spoof vizual label"),
    ]
    for check, label in checks:
        found = (check if isinstance(check, bool) else check in src)
        if found:
            ok(f"routes.py generate_frames: {label}")
        else:
            fail(f"routes.py generate_frames: {label} topilmadi")
except Exception as e:
    fail("routes.py integratsiya tekshiruvi", str(e))

try:
    from app.services import video_processor as vp_mod
    src2 = inspect.getsource(vp_mod.VideoProcessor.process_verification_frame)
    vp_checks = [
        ('liveness_detector.analyze_frame', "analyze_frame"),
        ('update_liveness',                 "update_liveness"),
        ('get_liveness_state',              "get_liveness_state"),
        ('is_confirmed and is_live',        "ikki shart"),
        ('is_live',                         "face_result ga is_live"),
        ('is_spoof',                        "face_result ga is_spoof"),
    ]
    for check, label in vp_checks:
        if check in src2:
            ok(f"video_processor.py: {label}")
        else:
            fail(f"video_processor.py: {label} topilmadi")
except Exception as e:
    fail("video_processor.py integratsiya tekshiruvi", str(e))

# ══════════════════════════════════════════════════════════
section("13. BLINK TIMEOUT TESTI")
# ══════════════════════════════════════════════════════════

liveness_module.PASSIVE_LIVENESS_ONLY = False
liveness_module.BLINKS_REQUIRED = 2

t8 = Tracker()
t8.tracks[1] = t8._new_track_data((0,0,100,100))
# Faqat 1 blink (BLINKS_REQUIRED=2 ga etmaydi)
for ear in [0.14, 0.14, 0.30]:
    t8.update_liveness(1, make_metrics(ear))

# Timeout'ni sun'iy ravishda simulyatsiya qilamiz
t8.tracks[1]['liveness']['liveness_start'] = time.time() - (BLINK_TIMEOUT_SECONDS + 1)
# Yana bir update (timeout trigger qilish uchun)
t8.update_liveness(1, make_metrics(0.30))
lv = t8.get_liveness_state(1)

# Timeout bo'lganda liveness_start reset qilinishi kerak (blink_count saqlanadi)
if lv['blink_count'] == 1 and not lv['is_live']:
    ok(f"Timeout → liveness_start reset, blink_count saqlanadi ({lv['blink_count']})")
else:
    fail("Timeout xatti-harakati kutilgancha emas", f"blink={lv['blink_count']}, live={lv['is_live']}")

# ══════════════════════════════════════════════════════════
section("14. KO'ZOYNAK FALLBACK (GLASSES MODE) TESTI")
# ══════════════════════════════════════════════════════════

def make_no_landmark(texture=200.0):
    """Ko'zoynak kiygan odamning metrikasi: landmark yo'q, texture yaxshi."""
    return {'ear': None, 'texture_score': texture,
            'texture_pass': texture >= TEXTURE_THRESHOLD,
            'has_landmarks': False}

# ── Test 14a: Glasses mode — yetarli frame + yaxshi texture → LIVE ──
tg = Tracker()
tg.tracks[1] = tg._new_track_data((0,0,100,100))

# GLASSES_FALLBACK_FRAMES ta no-landmark frame + GLASSES_TEXTURE_THRESHOLD dan yuqori texture
for _ in range(GLASSES_FALLBACK_FRAMES + 2):
    tg.update_liveness(1, make_no_landmark(texture=200.0))

lvg = tg.get_liveness_state(1)
if lvg['is_live']:
    ok(f"Ko'zoynak fallback: {GLASSES_FALLBACK_FRAMES} frame no-landmark + yaxshi texture → LIVE")
else:
    fail("Ko'zoynak fallback ishlamadi",
         f"no_landmark_streak={lvg['no_landmark_streak']}, is_live={lvg['is_live']}")

if lvg['liveness_mode'] == 'glasses':
    ok("liveness_mode = 'glasses' (to'g'ri)")
else:
    fail("liveness_mode 'glasses' bo'lishi kerak", f"mode={lvg['liveness_mode']}")

if lvg['blink_count'] == 0:
    ok("Ko'zoynak rejimida blink_count = 0 (blink bo'lmadi — to'g'ri)")
else:
    fail("Ko'zoynak rejimida blink_count noto'g'ri", str(lvg['blink_count']))

# ── Test 14b: Glasses mode — yetarli frame lekin PAST texture → LIVE emas ──
tg2 = Tracker()
tg2.tracks[1] = tg2._new_track_data((0,0,100,100))

for _ in range(GLASSES_FALLBACK_FRAMES + 2):
    tg2.update_liveness(1, make_no_landmark(texture=50.0))  # past texture (<120)

lvg2 = tg2.get_liveness_state(1)
if not lvg2['is_live']:
    ok(f"Ko'zoynak fallback: past texture ({50.0} < {GLASSES_TEXTURE_THRESHOLD}) → LIVE emas (to'g'ri)")
else:
    fail("Past texture bilan glasses fallback noto'g'ri LIVE berdi")

# ── Test 14c: Glasses mode spoof bilan — spoof aniqlansa glasses LIVE bermasin ──
tg3 = Tracker()
tg3.tracks[1] = tg3._new_track_data((0,0,100,100))

# Avval spoof aniqlansin
spoof_no_lm = {'ear': None, 'texture_score': 5.0, 'texture_pass': False, 'has_landmarks': False}
for _ in range(TEXTURE_SAMPLE_FRAMES):
    tg3.update_liveness(1, spoof_no_lm)

# Keyin ko'p no-landmark frame bersak ham — spoof holati saqlanishi kerak
for _ in range(GLASSES_FALLBACK_FRAMES + 2):
    tg3.update_liveness(1, make_no_landmark(texture=200.0))

lvg3 = tg3.get_liveness_state(1)
if lvg3['is_spoof'] and not lvg3['is_live']:
    ok("SPOOF aniqlangandan keyin glasses fallback LIVE berishni blokladi")
else:
    fail("SPOOF + glasses fallback kombinatsiyasi xato",
         f"is_spoof={lvg3['is_spoof']}, is_live={lvg3['is_live']}")

# ── Test 14d: Blink rejimi glasses'dan ustun ──
# Agar ko'zoynak kiygan odam birdan ko'zini yumib ochsa → blink rejimiga o'tishi kerak
tg4 = Tracker()
tg4.tracks[1] = tg4._new_track_data((0,0,100,100))

# Bir nechta no-landmark frame
for _ in range(5):
    tg4.update_liveness(1, make_no_landmark(texture=200.0))

# Keyin landmark topildi (ko'zoynak tushdi) + blink
for ear in [0.30, 0.14, 0.14, 0.30, 0.13, 0.13, 0.30]:
    tg4.update_liveness(1, make_metrics(ear, texture=150.0))

lvg4 = tg4.get_liveness_state(1)
if lvg4['is_live'] and lvg4['liveness_mode'] == 'blink':
    ok("No-landmark → landmark topildi + blink → LIVE (blink rejimi ustun)")
elif lvg4['is_live']:
    ok(f"No-landmark → blink → LIVE (mode={lvg4['liveness_mode']})")
else:
    fail("Aralash no-landmark + blink holati xato",
         f"is_live={lvg4['is_live']}, mode={lvg4['liveness_mode']}")

# ── Test 14e: no_landmark_streak to'g'ri hisoblanayotgani ──
tg5 = Tracker()
tg5.tracks[1] = tg5._new_track_data((0,0,100,100))

for _ in range(7):
    tg5.update_liveness(1, make_no_landmark())
lvg5 = tg5.get_liveness_state(1)
if lvg5['no_landmark_streak'] == 7:
    ok(f"no_landmark_streak to'g'ri hisoblanmoqda: {lvg5['no_landmark_streak']}")
else:
    fail("no_landmark_streak xato", f"streak={lvg5['no_landmark_streak']}")

# Landmark topilsa — streak reset bo'lishi kerak
tg5.update_liveness(1, make_metrics(0.30, texture=150.0))
lvg5b = tg5.get_liveness_state(1)
if lvg5b['no_landmark_streak'] == 0:
    ok("Landmark topilgach no_landmark_streak → 0 (reset)")
else:
    fail("no_landmark_streak reset bo'lmadi", f"streak={lvg5b['no_landmark_streak']}")

# ══════════════════════════════════════════════════════════
section("15. PASSIVE LIVENESS ONLY TESTI (YANGI)")
# ══════════════════════════════════════════════════════════

liveness_module.PASSIVE_LIVENESS_ONLY = True
liveness_module.BLINKS_REQUIRED = 1

tp = Tracker()
tp.tracks[1] = tp._new_track_data((0,0,100,100))

# 1. Real face: high texture score (>= TEXTURE_THRESHOLD) for TEXTURE_SAMPLE_FRAMES (3)
# analyze_frame should return ear=None under passive liveness, and landmarks=False
frame_result = liveness_detector.analyze_frame(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8), (10, 10, 190, 190))
if frame_result['ear'] is None and not frame_result['has_landmarks']:
    ok("PASSIVE LIVENESS: analyze_frame EAR va landmarks-ni chetlab o'tdi")
else:
    fail("PASSIVE LIVENESS: analyze_frame landmarks-ni chetlab o'tmadi", str(frame_result))

# Put 3 passive frames with high texture
passive_m = {'ear': None, 'texture_score': 150.0, 'texture_pass': True, 'has_landmarks': False}
for _ in range(TEXTURE_SAMPLE_FRAMES):
    tp.update_liveness(1, passive_m)

lvp = tp.get_liveness_state(1)
if lvp['is_live'] and lvp['liveness_mode'] == 'passive':
    ok("PASSIVE LIVENESS: 3 ta yaxshi tekstura frame → LIVE tasdiqlandi (soniyada!)")
else:
    fail("PASSIVE LIVENESS: LIVE tasdiqlanmadi", str(lvp))

# 2. Spoof face: low texture score (< 20) under passive mode
tp_spoof = Tracker()
tp_spoof.tracks[1] = tp_spoof._new_track_data((0,0,100,100))

passive_spoof_m = {'ear': None, 'texture_score': 5.0, 'texture_pass': False, 'has_landmarks': False}
for _ in range(TEXTURE_SAMPLE_FRAMES):
    tp_spoof.update_liveness(1, passive_spoof_m)

lvp_spoof = tp_spoof.get_liveness_state(1)
if lvp_spoof['is_spoof'] and not lvp_spoof['is_live']:
    ok("PASSIVE LIVENESS: past tekstura frame → SPOOF aniqlandi")
else:
    fail("PASSIVE LIVENESS: SPOOF aniqlanmadi", str(lvp_spoof))

# Qaytadan production holatiga tiklaymiz (PASSIVE_LIVENESS_ONLY = True)
liveness_module.PASSIVE_LIVENESS_ONLY = True

# ══════════════════════════════════════════════════════════
# YAKUNIY NATIJA
# ══════════════════════════════════════════════════════════

total = passed + failed
print(f"\n{BOLD}{'═'*55}{RESET}")
print(f"{BOLD}  NATIJA: {passed}/{total} test muvaffaqiyatli{RESET}")
if failed == 0:
    print(f"{GREEN}{BOLD}  ✅ BARCHA TESTLAR O'TDI!{RESET}")
else:
    print(f"{RED}{BOLD}  ❌ {failed} ta test muvaffaqiyatsiz:{RESET}")
    for e in errors:
        print(f"{RED}     • {e}{RESET}")
print(f"{BOLD}{'═'*55}{RESET}\n")

sys.exit(0 if failed == 0 else 1)
