import numpy as np
import time
import app.services.liveness as liveness_mod
from app.services.liveness import (
    EAR_THRESHOLD,
    EAR_CONSEC_FRAMES,
    BLINK_TIMEOUT_SECONDS,
    TEXTURE_SAMPLE_FRAMES,
    GLASSES_FALLBACK_FRAMES,
    GLASSES_TEXTURE_THRESHOLD,
    SPOOF_REGION_CV_MIN,
)

class Tracker:
    CONFIRM_THRESHOLD = 2  # Need 2 consecutive same-name results to confirm identity (reduced for speed)

    def __init__(self, max_lost=60, iou_threshold=0.4):
        """
        Args:
            max_lost: Frames before a track is deleted (increased from 15 to 60 for ~2 seconds)
            iou_threshold: Minimum IoU to consider a detection as same track
        """
        self.next_id = 1
        self.tracks = {}  # {id: track_data}
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.reverify_interval = 1.5  # Re-verify "Unknown" faces much faster (every 1.5s)
        self.unknown_retry_limit = 100  # Keep trying to recognize "Unknown" faces for a long time
        # Liveness parallel mode: start liveness even before full confirmation
        self.liveness_parallel = True

    def _new_liveness_state(self) -> dict:
        """Liveness tracking uchun yangi holat obyekti."""
        return {
            # Blink detection
            "blink_count":        0,      # Tasdiqlangan blinklar soni
            "ear_below_count":    0,      # Ketma-ket past EAR framelari (blink aniqlash uchun)
            "last_ear":           None,   # Oxirgi EAR qiymati (vizual uchun)

            # Texture / spoof
            "texture_scores":     [],     # So'nggi N ta texture balli
            "region_cvs":         [],     # So'nggi N ta regional CV balli
            "texture_pass":       False,  # Texture tekshiruvidan o'tdimi

            # Ko'zoynak fallback
            "no_landmark_streak": 0,     # Ketma-ket landmark topilmagan framelari
            "liveness_mode":      None,  # 'blink' | 'glasses' | None

            # Yakuniy holat
            "is_live":            False,  # Liveness to'liq tasdiqlangani
            "is_spoof":           False,  # Spoof aniqlangani (rasm/video)
            "liveness_start":     0.0,    # Kuzatuv boshlangan vaqt
        }

    def _new_track_data(self, bbox):
        """Create fresh track data with voting fields."""
        return {
            "bbox": bbox,
            "name": None,            # Final displayed name (None until confirmed or Unknown)
            "pending_name": None,     # Name being voted on
            "pending_votes": 0,       # How many consecutive times this name was returned
            "confirmed": False,       # Whether the identity has been confirmed
            "lost": 0,
            "last_verified": 0,
            "verify_count": 0,
            # Liveness holati (to'liq alohida dict)
            "liveness": self._new_liveness_state(),
        }

    def _calculate_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def update(self, detections):
        """
        detections: list of (x1, y1, x2, y2)
        returns: list of {"id": id, "bbox": bbox, "name": name, "is_new": bool,
                          "needs_reverify": bool, "is_confirmed": bool}
        """
        current_time = time.time()
        current_detections = []
        used_detections = set()
        
        # 1. Matching existing tracks
        for track_id, track_data in list(self.tracks.items()):
            best_iou = 0
            best_det_idx = -1
            
            for i, det in enumerate(detections):
                if i in used_detections:
                    continue
                iou = self._calculate_iou(track_data["bbox"], det)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_det_idx = i
            
            if best_det_idx != -1:
                # Track found!
                self.tracks[track_id]["bbox"] = detections[best_det_idx]
                self.tracks[track_id]["lost"] = 0
                used_detections.add(best_det_idx)
                
                # Determine if re-verification is needed
                needs_reverify = False
                name = self.tracks[track_id]["name"]
                confirmed = self.tracks[track_id].get("confirmed", False)
                last_verified = self.tracks[track_id].get("last_verified", 0)
                verify_count = self.tracks[track_id].get("verify_count", 0)
                
                # Case 1: Not yet confirmed — keep re-verifying rapidly
                if not confirmed and name != "Unknown":
                    if current_time - last_verified > 0.2:  # Re-verify every 0.2s for fast confirmation
                        needs_reverify = True

                # Case 2: Unknown face — retry periodically
                elif name == "Unknown" and verify_count < self.unknown_retry_limit:
                    if current_time - last_verified > self.reverify_interval:
                        needs_reverify = True
                
                current_detections.append({
                    "id": track_id, 
                    "bbox": detections[best_det_idx], 
                    "name": name,
                    "is_new": False,
                    "needs_reverify": needs_reverify,
                    "is_confirmed": confirmed,
                })
            else:
                # Track lost
                self.tracks[track_id]["lost"] += 1
        
        # 2. Add new tracks
        for i, det in enumerate(detections):
            if i not in used_detections:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = self._new_track_data(det)
                current_detections.append({
                    "id": new_id, 
                    "bbox": det, 
                    "name": None, 
                    "is_new": True,
                    "needs_reverify": False,
                    "is_confirmed": False,
                })
        
        # 3. Clean up old tracks
        ids_to_del = [tid for tid, data in self.tracks.items() if data["lost"] > self.max_lost]
        for tid in ids_to_del:
            del self.tracks[tid]
            
        return current_detections

    def set_name(self, track_id, name):
        """
        Set the recognized name for a track with multi-frame voting.
        
        The name is NOT immediately confirmed. It must be returned by the 
        recognizer CONFIRM_THRESHOLD times consecutively before it becomes
        confirmed (and safe to log attendance).
        
        Returns True if this call caused the identity to become CONFIRMED.
        """
        if track_id not in self.tracks:
            return False
        
        track = self.tracks[track_id]
        track["last_verified"] = time.time()
        track["verify_count"] = track.get("verify_count", 0) + 1
        
        # ── Unknown handling ──
        if name == "Unknown":
            # If we already have a pending candidate, DON'T reset it.
            # Intermittent bad frames shouldn't kill the voting process.
            if track.get("pending_name") and track.get("pending_votes", 0) > 0:
                # Keep the pending candidate alive — just skip this frame
                track["unknown_streak"] = track.get("unknown_streak", 0) + 1
                # But if too many consecutive Unknowns (5+), give up on the candidate
                if track["unknown_streak"] >= 5:
                    track["name"] = "Unknown"
                    track["pending_name"] = None
                    track["pending_votes"] = 0
                    track["confirmed"] = False
                    track["unknown_streak"] = 0
                return False
            else:
                # No pending candidate — just set Unknown
                track["name"] = "Unknown"
                track["pending_name"] = None
                track["pending_votes"] = 0
                track["confirmed"] = False
                track["unknown_streak"] = 0
                return False
        
        # ── Voting logic for known names ──
        track["unknown_streak"] = 0  # Reset unknown streak on any known result
        
        if name == track.get("pending_name"):
            # Same name as before — increase vote count
            track["pending_votes"] = track.get("pending_votes", 0) + 1
        else:
            # Different name — reset voting
            track["pending_name"] = name
            track["pending_votes"] = 1
        
        # Show the pending name on screen (for visual feedback)
        track["name"] = name
        
        # Check if confirmed
        if track["pending_votes"] >= self.CONFIRM_THRESHOLD:
            if not track.get("confirmed", False):
                track["confirmed"] = True
                print(f"[TRACKER] Identity CONFIRMED: Track {track_id} = {name} (after {track['pending_votes']} votes)")
                return True  # Just became confirmed
        else:
            track["confirmed"] = False
        
        return False
    
    def clear_name(self, track_id):
        """Clear the name to trigger re-verification"""
        if track_id in self.tracks:
            self.tracks[track_id]["name"] = None
            self.tracks[track_id]["pending_name"] = None
            self.tracks[track_id]["pending_votes"] = 0
            self.tracks[track_id]["confirmed"] = False
            self.tracks[track_id]["verify_count"] = 0

    def remove_user(self, name_to_remove):
        """Remove a user from all active tracks (e.g. after deletion)"""
        for tid, data in self.tracks.items():
            if data["name"] == name_to_remove:
                data["name"] = None
                data["pending_name"] = None
                data["pending_votes"] = 0
                data["confirmed"] = False
                data["verify_count"] = 0
                data["last_verified"] = 0
                # Liveness holatini ham reset qilish
                data["liveness"] = self._new_liveness_state()

    def reset_unknown_verifications(self):
        """Reset verification limit for all Unknown tracks to allow re-check after new registration"""
        for tid, data in self.tracks.items():
            if data["name"] == "Unknown":
                data["verify_count"] = 0
                data["last_verified"] = 0  # Force immediate retry

    # ── Liveness holati metodlari ──────────────────────────────────────────────

    def update_liveness(self, track_id: int, metrics: dict) -> None:
        """
        Liveness detector'dan kelgan metrikalar asosida track holatini yangilaydi.
        Blink detection va texture analysis natijalarini qayta ishlaydi.

        Args:
            track_id: Tracker ID
            metrics:  liveness_detector.analyze_frame() qaytargan dict
        """
        if track_id not in self.tracks:
            return

        lv = self.tracks[track_id]["liveness"]
        current_time = time.time()

        # Birinchi marta liveness kuzatuvi boshlanayotgan bo'lsa — vaqtni belgilaymiz
        if lv["liveness_start"] == 0.0:
            lv["liveness_start"] = current_time

        # ── Agar allaqachon xulosa chiqarilgan bo'lsa — ishlamaymiz ──
        if lv["is_live"] or lv["is_spoof"]:
            return

        # ── Qatlam 1: Texture / Spoof Detection ──────────────────────────────
        texture_score = metrics.get("texture_score", 0.0)
        region_cv     = metrics.get("region_cv", 1.0)   # Default 1.0 = o'tadi
        lv["texture_scores"].append(texture_score)
        lv["texture_scores"] = lv["texture_scores"][-TEXTURE_SAMPLE_FRAMES:]

        if "region_cvs" not in lv:
            lv["region_cvs"] = []
        lv["region_cvs"].append(region_cv)
        lv["region_cvs"] = lv["region_cvs"][-TEXTURE_SAMPLE_FRAMES:]

        if len(lv["texture_scores"]) >= TEXTURE_SAMPLE_FRAMES:
            avg_texture = sum(lv["texture_scores"]) / len(lv["texture_scores"])
            avg_cv      = sum(lv["region_cvs"]) / len(lv["region_cvs"])
            
            # Ikkala shart ham o'rtacha qiymatga ko'ra tekshiriladi
            lv["texture_pass"] = (avg_texture >= liveness_mod.TEXTURE_THRESHOLD) and (avg_cv >= liveness_mod.SPOOF_REGION_CV_MIN)

            # Juda past texture → spoof (bosma rasm, qorong'u)
            if avg_texture < 20.0:
                lv["is_spoof"] = True
                print(f"[TRACKER SPOOF] low avg_texture: {avg_texture:.2f}")
                return
            # Juda tekis regional texture → ekran rasmi yoki bosma qog'oz (spoof)
            if avg_cv < liveness_mod.SPOOF_REGION_CV_MIN:
                lv["is_spoof"] = True
                print(f"[TRACKER SPOOF] low avg_cv: {avg_cv:.4f} (avg_texture: {avg_texture:.2f})")
                return
        else:
            lv["texture_pass"] = False

        # ── PASSIVE ONLY MODE ──
        if liveness_mod.PASSIVE_LIVENESS_ONLY:
            # Ikkala shart: keskinlik VA regional notekislik (o'rtacha qiymatlar bo'yicha)
            if len(lv["texture_scores"]) >= TEXTURE_SAMPLE_FRAMES and lv["texture_pass"]:
                lv["is_live"] = True
                lv["liveness_mode"] = "passive"
            return

        # ── Qatlam 2: Blink Detection (EAR) ──────────────────────────────────
        ear = metrics.get("ear")
        has_landmarks = metrics.get("has_landmarks", False)

        if ear is not None and has_landmarks:
            # Landmark topildi — blink detection ishlaydi
            lv["no_landmark_streak"] = 0   # streak'ni reset qilamiz
            lv["last_ear"] = ear

            # Dinamik EAR chegarasi boshlang'ich qiymati
            if "max_ear" not in lv or lv["max_ear"] is None:
                lv["max_ear"] = ear
            else:
                # Ko'z ochiq paytdagi eng yuqori EAR qiymatini saqlab boramiz (shovqindan holi 0.45 gacha)
                if ear > lv["max_ear"] and ear < 0.45:
                    lv["max_ear"] = ear

            # Dinamik EAR chegarasi: max_ear ning 72% qismi, [0.16, 0.24] diapazonida cheklangan
            dynamic_threshold = max(0.16, min(0.24, 0.72 * lv["max_ear"]))

            if ear < dynamic_threshold:
                # Ko'z yopilmoqda
                lv["ear_below_count"] += 1
            else:
                # Ko'z ochildi — agar yetarli konsekutiv frame bo'lsa → blink
                if lv["ear_below_count"] >= EAR_CONSEC_FRAMES:
                    lv["blink_count"] += 1
                    print(f"[TRACKER BLINK] Blink registered! Total count: {lv['blink_count']} (EAR: {ear:.3f}, Dynamic Threshold: {dynamic_threshold:.3f}, Max EAR: {lv['max_ear']:.3f})")
                lv["ear_below_count"] = 0
        else:
            # Landmark topilmadi (ko'zoynak, burchak, past yorug'lik)
            lv["no_landmark_streak"] = lv.get("no_landmark_streak", 0) + 1

        # ── Qatlam 3: Ko'zoynak Fallback (Glasses Mode) ───────────────────────
        # Landmark uzoq vaqt topilmasa lekin texture barqaror yaxshi bo'lsa → LIVE
        # Commented out to prevent static background elements from triggering "live" status.
        # no_lm = lv["no_landmark_streak"]
        # if no_lm >= GLASSES_FALLBACK_FRAMES and len(lv["texture_scores"]) >= TEXTURE_SAMPLE_FRAMES:
        #     avg_texture = sum(lv["texture_scores"]) / len(lv["texture_scores"])
        #     if avg_texture >= GLASSES_TEXTURE_THRESHOLD:
        #         lv["is_live"]       = True
        #         lv["liveness_mode"] = "glasses"
        #         return  # Glasses rejimi orqali tasdiqlandi

        # ── Timeout tekshiruvi ────────────────────────────────────────────────
        elapsed = current_time - lv["liveness_start"]
        if elapsed > BLINK_TIMEOUT_SECONDS and not lv["is_live"]:
            if lv["blink_count"] < liveness_mod.BLINKS_REQUIRED:
                # Qayta urinish: davomiylikni reset qilamiz, lekin blink_count saqlaymiz
                lv["liveness_start"] = current_time

        # ── Xulosa: LIVE (blink rejimi) ───────────────────────────────────────
        # Faqat ko'z pirpiratilishi kifoya
        if lv["blink_count"] >= liveness_mod.BLINKS_REQUIRED:
            lv["is_live"]       = True
            lv["liveness_mode"] = "blink"

    def get_liveness_state(self, track_id: int) -> dict:
        """
        Track uchun joriy liveness holati.

        Returns:
            dict: {
                'is_live':       bool,
                'is_spoof':      bool,
                'blink_count':   int,
                'texture_pass':  bool,
                'last_ear':      float | None,
                'liveness_mode': str | None,   # 'blink' | 'glasses' | None
                'no_landmark_streak': int,     # Landmark topilmagan framelari
            }
        """
        if track_id not in self.tracks:
            return {"is_live": False, "is_spoof": False, "blink_count": 0,
                    "texture_pass": False, "last_ear": None,
                    "liveness_mode": None, "no_landmark_streak": 0}
        lv = self.tracks[track_id]["liveness"]
        return {
            "is_live":            lv["is_live"],
            "is_spoof":           lv["is_spoof"],
            "blink_count":        lv["blink_count"],
            "texture_pass":       lv["texture_pass"],
            "last_ear":           lv["last_ear"],
            "liveness_mode":      lv.get("liveness_mode"),
            "no_landmark_streak": lv.get("no_landmark_streak", 0),
        }
