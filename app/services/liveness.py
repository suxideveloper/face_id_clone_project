"""
Liveness Detection Service
==========================
Uch qatlamli haqiqiy yuz tekshiruvi:

  Qatlam 1 — Texture / Spoof Detection (Passive, Instant)
    Laplacian variance orqali yuz ROI-ning keskinligi hisoblanadi.
    Bosib chiqarilgan rasm yoki ekrandagi foto past keskinlik (blur) beradi.
    Threshold: variance < 20 → SPOOF (darhol rad etish)

  Qatlam 2 — Blink Detection (Active, 3-5 soniya) [ASOSIY]
    dlib 68-point landmark modeli bilan ko'z holatini kuzatib,
    Eye Aspect Ratio (EAR) yordamida ko'z pirpirashlari aniqlanadi.
    Rasmda ko'z hech qachon pirpirmaydi.
    Threshold: blink_count >= BLINKS_REQUIRED → LIVE

  Qatlam 3 — Texture-Only Fallback (Ko'zoynak kiyganlar uchun)
    Ko'zoynak yoki yuz burchagi sababli landmark aniqlanmasa:
    GLASSES_FALLBACK_FRAMES ta ketma-ket frame landmark topilmasa
    lekin texture barqaror yaxshi bo'lsa → LIVE (texture-only rejim).
    Bu rejim blink detection'dan kamroq ishonchli, shuning uchun
    texture sifatiga qo'yilgan talablar qat'iyroq.

Best Practices:
  - Singleton pattern (modul darajasida bitta instance)
  - Stateless analyze_frame() — holat Tracker'da saqlanadi
  - Xatoliklarni yutmaslik: har bir xato loglanadi
  - Magic numberlar konstantalar sifatida
"""

import cv2
import numpy as np
import dlib
import face_recognition_models
import logging

logger = logging.getLogger(__name__)


# ── Sozlanuvchi konstantalar ───────────────────────────────────────────────────
PASSIVE_LIVENESS_ONLY = True  # Faqat passive tekstura tahlili ishlatilsin (dlib va active blink o'chiriladi)

# Blink Detection (EAR) - Faqat PASSIVE_LIVENESS_ONLY = False bo'lganda ishlaydi
EAR_THRESHOLD = 0.21          # Qo'z yopiq: EAR < bu qiymat
EAR_CONSEC_FRAMES = 2         # Necha consecutive frame past EAR → bir blink
BLINKS_REQUIRED = 1           # Liveness tasdiqlash uchun kerakli blink soni (1 ta yetarli)
BLINK_TIMEOUT_SECONDS = 5.0   # Shu vaqt ichida blink bo'lmasa → fail (8s o'rniga 5s)

# Texture / Spoof Detection
TEXTURE_THRESHOLD = 60.0      # Laplacian variance. Pastroq = xiralash = rasm
TEXTURE_SAMPLE_FRAMES = 3     # Necha frameda tekshirish (noto'g'ri ijobiyni kamaytiradi)

# Ko'zoynak Fallback (Glasses Mode) - Faqat PASSIVE_LIVENESS_ONLY = False bo'lganda ishlaydi
# Landmark aniqlanmasa shu qadar ketma-ket frame o'tsa → texture-only rejimga o'tish
GLASSES_FALLBACK_FRAMES = 12  # ≈ 28FPS da ~0.4 soniya (20 o'rniga 12 — tezroq)
# Texture-only rejim uchun qat'iyroq threshold (blink yo'q, texture ishonchliroq bo'lishi kerak)
GLASSES_TEXTURE_THRESHOLD = 100.0  # Odatiy 60.0 dan kattaroq (120.0 o'rniga 100.0)

# Landmark indekslari (dlib 68-point)
LEFT_EYE_INDICES  = list(range(36, 42))   # 6 ta nuqta: chapki ko'z
RIGHT_EYE_INDICES = list(range(42, 48))   # 6 ta nuqta: o'ng ko'z


class LivenessDetector:
    """
    Stateless liveness analyzer.
    Holat (blink_count, ear_history va h.k.) tashqi Tracker'da saqlanadi.
    Bu sinf faqat bitta frame uchun metrikalarni hisoblaydi.
    """

    def __init__(self):
        if PASSIVE_LIVENESS_ONLY:
            self._dlib_detector = None
            self._predictor = None
            logger.info("LivenessDetector tayyor: PASSIVE_LIVENESS_ONLY yoqilgan, dlib yuklanmadi.")
        else:
            try:
                # dlib face detector (landmark topish uchun)
                self._dlib_detector = dlib.get_frontal_face_detector()

                # 68-point shape predictor (face_recognition kutubxonasining modeli)
                model_path = face_recognition_models.pose_predictor_model_location()
                self._predictor = dlib.shape_predictor(model_path)
                logger.info("LivenessDetector tayyor: 68-point model yuklandi.")
            except Exception as e:
                logger.error("dlib modellarini yuklashda xatolik: %s. Passive rejimga majburiy o'tiladi.", e)
                self._dlib_detector = None
                self._predictor = None

    # ── EAR (Eye Aspect Ratio) ─────────────────────────────────────────────────

    def _get_ear(self, eye_points: np.ndarray) -> float:
        """
        EAR formulasi:
            EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        Ko'z ochiq: EAR ≈ 0.25-0.30
        Ko'z yopiq: EAR < 0.21

        Args:
            eye_points: (6, 2) shape — ko'z uchun 6 ta (x, y) koordinata

        Returns:
            float: EAR qiymati, [0.0, 1.0] oraliqda
        """
        # Vertikal masofalar
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        # Gorizontal masofa
        h  = np.linalg.norm(eye_points[0] - eye_points[3])
        if h < 1e-6:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def _landmarks_to_np(self, shape) -> np.ndarray:
        """dlib shape objektini (68, 2) numpy array ga o'zgartiradi."""
        coords = np.zeros((68, 2), dtype=np.float64)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def _get_ear_from_frame(
        self,
        gray_frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> float | None:
        """
        Berilgan bbox ichidagi yuz uchun o'rtacha EAR hisoblaydi.

        Args:
            gray_frame: Grayscale frame (butun frame, crop emas)
            bbox: (x1, y1, x2, y2) — yuz chegarasi

        Returns:
            float | None: EAR qiymati, yoki None (landmark topilmasa)
        """
        if PASSIVE_LIVENESS_ONLY or self._predictor is None:
            return None

        x1, y1, x2, y2 = bbox
        # dlib rectangle (sol, yuqori, o'ng, past) tartibida
        rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)

        try:
            shape = self._predictor(gray_frame, rect)
        except Exception as exc:
            logger.debug("Landmark predictor xatosi: %s", exc)
            return None

        lm = self._landmarks_to_np(shape)
        left_ear  = self._get_ear(lm[LEFT_EYE_INDICES])
        right_ear = self._get_ear(lm[RIGHT_EYE_INDICES])
        return (left_ear + right_ear) / 2.0

    # ── Texture / Spoof Detection ──────────────────────────────────────────────

    def _check_texture(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        """
        Yuz ROI uchun Laplacian variance hisoblaydi.

        Haqiqiy yuz: terining mikroteksturasi yuqori keskinlik beradi (> TEXTURE_THRESHOLD).
        Bosib chiqarilgan rasm / ekran: past keskinlik (xiralash, tekis piksellar).

        Args:
            frame: BGR frame
            bbox: (x1, y1, x2, y2)

        Returns:
            float: variance qiymati (yuqori = real, past = spoof)
        """
        x1, y1, x2, y2 = bbox
        # Crop va resize (standart o'lcham uchun)
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return 0.0

        # Standart o'lchamga keltirish (hisob barqarorligi uchun)
        face_roi = cv2.resize(face_roi, (64, 64))
        gray     = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Laplacian (ikkinchi tartibli hosilaning variansasi = keskinlik o'lchami)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    # ── Asosiy tahlil metodi ───────────────────────────────────────────────────

    def analyze_frame(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> dict:
        """
        Bitta frame uchun liveness metrikalarini hisoblaydi.
        Bu metod stateless — holatni Tracker saqlab boradi.

        Args:
            frame: BGR numpy array (butun kamera frame)
            bbox: (x1, y1, x2, y2) — aniqlangan yuz chegarasi

        Returns:
            dict: {
                'ear': float | None,           # Ko'z EAR qiymati
                'texture_score': float,        # Laplacian variance
                'texture_pass': bool,          # Texture tekshiruvi o'tdimi
                'has_landmarks': bool,         # Landmark topildimi
            }
        """
        # 1. EAR hisoblash (faqat passive bo'lmaganda)
        if PASSIVE_LIVENESS_ONLY:
            ear = None
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ear = self._get_ear_from_frame(gray, bbox)

        # 2. Texture hisoblash
        texture_score = self._check_texture(frame, bbox)
        texture_pass  = texture_score >= TEXTURE_THRESHOLD

        return {
            "ear":           ear,
            "texture_score": texture_score,
            "texture_pass":  texture_pass,
            "has_landmarks": ear is not None,
        }


# ── Singleton instance ─────────────────────────────────────────────────────────
liveness_detector = LivenessDetector()
