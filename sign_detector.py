import os
import math
import time
import json
import sqlite3
import urllib.request
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gestures.db")

# Standard 21 hand landmark connections for skeleton drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky & Palm
]

# Built-in Sign Language Dictionary (Common Communication Signs)
SIGN_DICTIONARY: Dict[str, Dict[str, Any]] = {
    "hello": {
        "id": "hello",
        "label_vi": "Xin chào",
        "label_en": "Hello",
        "emoji": "👋",
        "category": "Chào hỏi",
        "how_to": "Giơ bàn tay mở rộng, xòe cả 5 ngón tay hướng về phía camera.",
        "color": "#3b82f6"
    },
    "thank_you": {
        "id": "thank_you",
        "label_vi": "Cảm ơn",
        "label_en": "Thank You",
        "emoji": "🙏",
        "category": "Giao tiếp",
        "how_to": "Duỗi thẳng các ngón tay và khép sát 4 ngón (trỏ, giữa, áp út, út) vào nhau.",
        "color": "#10b981"
    },
    "i_love_you": {
        "id": "i_love_you",
        "label_vi": "Tôi yêu bạn",
        "label_en": "I Love You",
        "emoji": "🤟",
        "category": "Cảm xúc",
        "how_to": "Duỗi thẳng ngón cái, ngón trỏ và ngón út; gập ngón giữa và ngón áp út xuống.",
        "color": "#ec4899"
    },
    "yes_good": {
        "id": "yes_good",
        "label_vi": "Đồng ý / Tốt",
        "label_en": "Yes / Good",
        "emoji": "👍",
        "category": "Phản hồi",
        "how_to": "Nắm 4 ngón tay lại và giơ ngón cái hướng thẳng lên trên.",
        "color": "#22c55e"
    },
    "no_bad": {
        "id": "no_bad",
        "label_vi": "Không đồng ý",
        "label_en": "No / Dislike",
        "emoji": "👎",
        "category": "Phản hồi",
        "how_to": "Nắm 4 ngón tay lại và chúc ngón cái hướng xuống dưới.",
        "color": "#ef4444"
    },
    "ok": {
        "id": "ok",
        "label_vi": "OK / Ổn",
        "label_en": "OK / Fine",
        "emoji": "👌",
        "category": "Phản hồi",
        "how_to": "Chạm đầu ngón cái và đầu ngón trỏ tạo thành vòng tròn, duỗi thẳng 3 ngón còn lại.",
        "color": "#06b6d4"
    },
    "sorry": {
        "id": "sorry",
        "label_vi": "Xin lỗi",
        "label_en": "Sorry",
        "emoji": "✊",
        "category": "Giao tiếp",
        "how_to": "Nắm gọn cả 5 ngón tay thành nắm đấm hướng về phía trước.",
        "color": "#f59e0b"
    },
    "peace": {
        "id": "peace",
        "label_vi": "Tạm biệt / Hòa bình",
        "label_en": "Peace / Bye",
        "emoji": "✌️",
        "category": "Chào hỏi",
        "how_to": "Giơ ngón trỏ và ngón giữa tạo thành hình chữ V, các ngón còn lại gập xuống.",
        "color": "#8b5cf6"
    },
    "call_me": {
        "id": "call_me",
        "label_vi": "Gọi cho tôi",
        "label_en": "Call Me",
        "emoji": "🤙",
        "category": "Giao tiếp",
        "how_to": "Duỗi ngón cái và ngón út ra ngoài, gập 3 ngón giữa (trỏ, giữa, áp út) vào lòng bàn tay.",
        "color": "#14b8a6"
    },
    "point_one": {
        "id": "point_one",
        "label_vi": "Tôi / Chú ý",
        "label_en": "Me / Attention",
        "emoji": "☝️",
        "category": "Giao tiếp",
        "how_to": "Chỉ duỗi thẳng ngón trỏ hướng lên trên, gập 4 ngón còn lại.",
        "color": "#6366f1"
    },
    "help": {
        "id": "help",
        "label_vi": "Cần giúp đỡ",
        "label_en": "Help Me",
        "emoji": "🆘",
        "category": "Khẩn cấp",
        "how_to": "Giơ 3 ngón giữa (trỏ, giữa, áp út) thẳng lên (chữ W) hoặc kết hợp 2 bàn tay (1 tay nắm + 1 tay xòe).",
        "color": "#f43f5e"
    },
    "stop": {
        "id": "stop",
        "label_vi": "Dừng lại",
        "label_en": "Stop / Wait",
        "emoji": "✋",
        "category": "Khẩn cấp",
        "how_to": "Duỗi thẳng 4 ngón tay (trỏ, giữa, áp út, út) và gập ngón cái vào trong lòng bàn tay.",
        "color": "#eab308"
    }
}


def ensure_model_exists() -> str:
    """Ensure the MediaPipe HandLandmarker model file exists locally."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000:
        print(f"Downloading MediaPipe HandLandmarker model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model download complete.")
    return MODEL_PATH


def _dist2d(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _dist3d(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.sqrt(
        (a["x"] - b["x"]) ** 2 +
        (a["y"] - b["y"]) ** 2 +
        (a["z"] - b["z"]) ** 2
    )


def _angle_deg(a: Dict[str, float], b: Dict[str, float], c: Dict[str, float]) -> float:
    """Compute angle ABC in degrees in 2D/3D."""
    v1 = (a["x"] - b["x"], a["y"] - b["y"], a.get("z", 0.0) - b.get("z", 0.0))
    v2 = (c["x"] - b["x"], c["y"] - b["y"], c.get("z", 0.0) - b.get("z", 0.0))
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    n1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) + 1e-8
    n2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2) + 1e-8
    cos_val = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_val))


def normalize_landmarks(landmarks: List[Dict[str, float]]) -> List[float]:
    """Convert 21 landmarks into a scale- and translation-invariant 42D feature vector."""
    wrist = landmarks[0]
    palm_size = _dist2d(landmarks[0], landmarks[9]) + 1e-6
    features: List[float] = []
    for lm in landmarks:
        features.append((lm["x"] - wrist["x"]) / palm_size)
        features.append((lm["y"] - wrist["y"]) / palm_size)
    return features


class SignLanguageDetector:
    def __init__(self) -> None:
        model_path = ensure_model_exists()
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.init_db()
        self.custom_signs = self.load_custom_signs()

    def init_db(self) -> None:
        """Initialize SQLite tables for custom gestures & detection logs."""
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS custom_signs (
                    id TEXT PRIMARY KEY,
                    label_vi TEXT NOT NULL,
                    label_en TEXT NOT NULL,
                    emoji TEXT DEFAULT '🖐️',
                    category TEXT DEFAULT 'Tùy chỉnh',
                    how_to TEXT DEFAULT 'Cử chỉ do người dùng tự huấn luyện',
                    color TEXT DEFAULT '#a855f7',
                    feature_vector TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sign_id TEXT NOT NULL,
                    label_vi TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def load_custom_signs(self) -> List[Dict[str, Any]]:
        """Load user-trained custom sign templates from SQLite."""
        try:
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, label_vi, label_en, emoji, category, how_to, color, feature_vector FROM custom_signs"
                )
                rows = cursor.fetchall()
                result = []
                for r in rows:
                    result.append({
                        "id": r[0],
                        "label_vi": r[1],
                        "label_en": r[2],
                        "emoji": r[3],
                        "category": r[4],
                        "how_to": r[5],
                        "color": r[6],
                        "feature_vector": json.loads(r[7]),
                        "is_custom": True
                    })
                return result
        except Exception as e:
            print(f"Error loading custom signs: {e}")
            return []

    def save_custom_sign(
        self,
        sign_id: str,
        label_vi: str,
        label_en: str,
        emoji: str,
        how_to: str,
        landmarks: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Save a custom sign template from a 21-landmark hand sample."""
        features = normalize_landmarks(landmarks)
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO custom_signs (id, label_vi, label_en, emoji, category, how_to, color, feature_vector)
                VALUES (?, ?, ?, ?, 'Tùy chỉnh', ?, '#a855f7', ?)
                ON CONFLICT(id) DO UPDATE SET
                    label_vi = excluded.label_vi,
                    label_en = excluded.label_en,
                    emoji = excluded.emoji,
                    how_to = excluded.how_to,
                    feature_vector = excluded.feature_vector
            """, (sign_id, label_vi, label_en, emoji or "🖐️", how_to or "Ký hiệu tùy chỉnh", json.dumps(features)))
            conn.commit()
        self.custom_signs = self.load_custom_signs()
        return {
            "id": sign_id,
            "label_vi": label_vi,
            "label_en": label_en,
            "emoji": emoji or "🖐️",
            "category": "Tùy chỉnh",
            "how_to": how_to or "Ký hiệu tùy chỉnh",
            "color": "#a855f7",
            "is_custom": True
        }

    def delete_custom_sign(self, sign_id: str) -> bool:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM custom_signs WHERE id = ?", (sign_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
        self.custom_signs = self.load_custom_signs()
        return deleted

    def get_all_signs(self) -> List[Dict[str, Any]]:
        builtin = [{**v, "is_custom": False} for v in SIGN_DICTIONARY.values()]
        custom = [
            {k: v for k, v in s.items() if k != "feature_vector"}
            for s in self.custom_signs
        ]
        return builtin + custom

    def analyze_fingers(self, lm: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Determine extension state of each finger in a rotation-aware, scale-invariant way.
        Landmarks:
        0: WRIST
        1-4: THUMB (CMC, MCP, IP, TIP)
        5-8: INDEX (MCP, PIP, DIP, TIP)
        9-12: MIDDLE (MCP, PIP, DIP, TIP)
        13-16: RING (MCP, PIP, DIP, TIP)
        17-20: PINKY (MCP, PIP, DIP, TIP)
        """
        wrist = lm[0]
        palm_width = _dist2d(lm[5], lm[17]) + 1e-6
        palm_length = _dist2d(lm[0], lm[9]) + 1e-6
        scale = (palm_width + palm_length) * 0.5

        # For the 4 fingers (Index, Middle, Ring, Pinky):
        # A finger is extended if:
        # 1. Distance from wrist to TIP > distance from wrist to PIP * 1.12
        # 2. The joint angle at PIP is relatively straight (> 135 degrees)
        def is_finger_extended(mcp_idx: int, pip_idx: int, dip_idx: int, tip_idx: int) -> bool:
            dist_tip_wrist = _dist2d(lm[tip_idx], wrist)
            dist_pip_wrist = _dist2d(lm[pip_idx], wrist)
            dist_mcp_wrist = _dist2d(lm[mcp_idx], wrist)
            pip_angle = _angle_deg(lm[mcp_idx], lm[pip_idx], lm[tip_idx])
            return (
                dist_tip_wrist > dist_pip_wrist * 1.08
                and dist_tip_wrist > dist_mcp_wrist * 1.25
                and pip_angle > 130.0
            )

        index_up = is_finger_extended(5, 6, 7, 8)
        middle_up = is_finger_extended(9, 10, 11, 12)
        ring_up = is_finger_extended(13, 14, 15, 16)
        pinky_up = is_finger_extended(17, 18, 19, 20)

        # Thumb extension check:
        # Check angle at thumb IP/MCP and distance from thumb tip (4) to index MCP (5) and pinky MCP (17)
        thumb_angle = _angle_deg(lm[1], lm[2], lm[4])
        thumb_ip_angle = _angle_deg(lm[2], lm[3], lm[4])
        dist_thumb_pinky_mcp = _dist2d(lm[4], lm[17]) / palm_width
        dist_thumb_index_mcp = _dist2d(lm[4], lm[5]) / palm_width
        dist_thumb_middle_pip = _dist2d(lm[4], lm[10]) / palm_width

        thumb_up = (
            thumb_angle > 135.0
            and thumb_ip_angle > 135.0
            and dist_thumb_pinky_mcp > 1.05
            and dist_thumb_index_mcp > 0.55
            and dist_thumb_middle_pip > 0.50
        )

        return {
            "thumb": thumb_up,
            "index": index_up,
            "middle": middle_up,
            "ring": ring_up,
            "pinky": pinky_up,
            "scale": scale,
            "palm_width": palm_width,
            "palm_length": palm_length,
        }

    def classify_single_hand(
        self, lm: List[Dict[str, float]], finger_info: Dict[str, Any]
    ) -> Tuple[Optional[str], float]:
        """Classify a single hand's 21 landmarks into a sign ID and confidence score."""
        # 1. Check custom user-trained signs first using normalized Euclidean distance
        if self.custom_signs:
            feat = normalize_landmarks(lm)
            best_custom_id = None
            best_dist = 999.0
            for csign in self.custom_signs:
                target = csign["feature_vector"]
                if len(target) == len(feat):
                    d = math.sqrt(sum((a - b) ** 2 for a, b in zip(feat, target)) / len(feat))
                    if d < best_dist:
                        best_dist = d
                        best_custom_id = csign["id"]
            if best_custom_id is not None and best_dist < 0.22:
                conf = max(0.75, min(0.99, 1.0 - (best_dist / 0.30)))
                return best_custom_id, round(conf, 2)

        thumb = finger_info["thumb"]
        index = finger_info["index"]
        middle = finger_info["middle"]
        ring = finger_info["ring"]
        pinky = finger_info["pinky"]
        palm_width = finger_info["palm_width"]

        # Check OK sign: Thumb tip (4) and Index tip (8) are very close, and middle + ring + pinky are extended
        thumb_index_dist = _dist2d(lm[4], lm[8]) / palm_width
        if thumb_index_dist < 0.42 and middle and ring and pinky:
            conf = min(0.98, max(0.82, 1.0 - thumb_index_dist * 0.4))
            return "ok", round(conf, 2)

        # Check I Love You (🤟): Thumb + Index + Pinky extended, Middle + Ring curled
        if thumb and index and (not middle) and (not ring) and pinky:
            return "i_love_you", 0.96

        # Check Call Me (🤙): Thumb + Pinky extended, Index + Middle + Ring curled
        if thumb and (not index) and (not middle) and (not ring) and pinky:
            return "call_me", 0.94

        # Check All 4 main fingers extended (Index, Middle, Ring, Pinky)
        if index and middle and ring and pinky:
            # Measure finger spread between adjacent fingertips normalized by palm width
            spread = (
                _dist2d(lm[8], lm[12])
                + _dist2d(lm[12], lm[16])
                + _dist2d(lm[16], lm[20])
            ) / palm_width

            if not thumb:
                # 4 fingers straight, thumb tucked in -> Stop / Wait (✋) or Thank You
                if spread < 0.95:
                    return "thank_you", 0.91
                return "stop", 0.90

            # All 5 fingers extended: distinguish "Xin chào" (fingers spread) vs "Cảm ơn" (fingers held close together)
            if spread >= 0.98:
                conf = min(0.98, 0.85 + (spread - 0.98) * 0.25)
                return "hello", round(conf, 2)
            else:
                conf = min(0.96, 0.85 + (0.98 - spread) * 0.25)
                return "thank_you", round(conf, 2)

        # Check 3 fingers up (Index + Middle + Ring extended, Pinky + Thumb curled) -> Help (🆘)
        if index and middle and ring and (not pinky):
            return "help", 0.91

        # Check Peace / Bye (✌️): Index + Middle extended, Ring + Pinky curled
        if index and middle and (not ring) and (not pinky):
            return "peace", 0.95

        # Check Point / Attention (☝️): Only Index extended
        if index and (not middle) and (not ring) and (not pinky) and (not thumb):
            return "point_one", 0.93

        # Check Thumbs Up (👍) vs Thumbs Down (👎) vs Fist / Sorry (✊)
        if (not index) and (not middle) and (not ring) and (not pinky):
            # Check if thumb is clearly pointing up or down relative to wrist and index MCP
            dy_wrist = lm[4]["y"] - lm[0]["y"]
            dy_mcp = lm[4]["y"] - lm[5]["y"]
            thumb_reach = _dist2d(lm[4], lm[9]) / palm_width

            if (thumb or thumb_reach > 0.65) and dy_mcp < -0.25 * palm_width and dy_wrist < -0.20 * palm_width:
                return "yes_good", 0.95
            if (thumb or thumb_reach > 0.65) and dy_mcp > 0.25 * palm_width and dy_wrist > 0.15 * palm_width:
                return "no_bad", 0.94

            # Closed fist -> Sorry (✊)
            return "sorry", 0.90

        return None, 0.0

    def process_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """Decode image bytes, run MediaPipe HandLandmarker, and return structured sign recognition result."""
        t0 = time.perf_counter()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return {"error": "Invalid image data", "hands": [], "detected_sign": None}

        return self.process_bgr_frame(frame_bgr, t0)

    def process_bgr_frame(self, frame_bgr: np.ndarray, t0: Optional[float] = None) -> Dict[str, Any]:
        if t0 is None:
            t0 = time.perf_counter()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect(mp_image)

        hands_output: List[Dict[str, Any]] = []
        single_hand_predictions: List[Tuple[str, float]] = []

        if result.hand_landmarks:
            for idx, hand_lms in enumerate(result.hand_landmarks):
                handedness_label = "Right"
                handedness_score = 0.95
                if result.handedness and idx < len(result.handedness) and len(result.handedness[idx]) > 0:
                    handedness_label = result.handedness[idx][0].category_name
                    handedness_score = round(float(result.handedness[idx][0].score), 2)

                lm_list = [
                    {
                        "x": round(float(pt.x), 4),
                        "y": round(float(pt.y), 4),
                        "z": round(float(pt.z), 4),
                    }
                    for pt in hand_lms
                ]

                finger_info = self.analyze_fingers(lm_list)
                sign_id, conf = self.classify_single_hand(lm_list, finger_info)
                if sign_id:
                    single_hand_predictions.append((sign_id, conf))

                hands_output.append({
                    "handedness": handedness_label,
                    "score": handedness_score,
                    "landmarks": lm_list,
                    "fingers": {
                        "thumb": finger_info["thumb"],
                        "index": finger_info["index"],
                        "middle": finger_info["middle"],
                        "ring": finger_info["ring"],
                        "pinky": finger_info["pinky"],
                    },
                    "sign_id": sign_id,
                    "confidence": conf,
                })

        # Two-hand combination check:
        # In Sign Language, 1 Open Palm ("hello"/"thank_you") + 1 Fist/Thumbs-Up ("sorry"/"yes_good") = "Help" (Cần giúp đỡ)
        final_sign_id: Optional[str] = None
        final_confidence: float = 0.0

        if len(single_hand_predictions) == 2:
            s1, c1 = single_hand_predictions[0]
            s2, c2 = single_hand_predictions[1]
            pair = {s1, s2}
            if pair.intersection({"hello", "thank_you", "stop"}) and pair.intersection({"sorry", "yes_good"}):
                final_sign_id = "help"
                final_confidence = 0.97

        if final_sign_id is None and single_hand_predictions:
            # Pick prediction with highest confidence
            single_hand_predictions.sort(key=lambda x: x[1], reverse=True)
            final_sign_id, final_confidence = single_hand_predictions[0]

        detected_sign_meta = None
        if final_sign_id:
            if final_sign_id in SIGN_DICTIONARY:
                detected_sign_meta = {
                    **SIGN_DICTIONARY[final_sign_id],
                    "confidence": final_confidence,
                    "is_custom": False,
                }
            else:
                for csign in self.custom_signs:
                    if csign["id"] == final_sign_id:
                        detected_sign_meta = {
                            k: v for k, v in csign.items() if k != "feature_vector"
                        }
                        detected_sign_meta["confidence"] = final_confidence
                        break

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        return {
            "hands": hands_output,
            "connections": HAND_CONNECTIONS,
            "detected_sign": detected_sign_meta,
            "latency_ms": latency_ms,
        }

    def process_landmarks_payload(self, hands_input: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process pre-extracted or simulated 21-point hand landmarks through the Python
        geometric finger analyzer and sign classifier (for camera-free testing).
        """
        t0 = time.perf_counter()
        hands_output: List[Dict[str, Any]] = []
        single_hand_predictions: List[Tuple[str, float]] = []

        for hand_item in hands_input:
            lm_list = hand_item.get("landmarks", [])
            if len(lm_list) != 21:
                continue
            handedness_label = hand_item.get("handedness", "Right")
            handedness_score = float(hand_item.get("score", 0.98))

            finger_info = self.analyze_fingers(lm_list)
            sign_id, conf = self.classify_single_hand(lm_list, finger_info)
            if sign_id:
                single_hand_predictions.append((sign_id, conf))

            hands_output.append({
                "handedness": handedness_label,
                "score": handedness_score,
                "landmarks": lm_list,
                "fingers": {
                    "thumb": finger_info["thumb"],
                    "index": finger_info["index"],
                    "middle": finger_info["middle"],
                    "ring": finger_info["ring"],
                    "pinky": finger_info["pinky"],
                },
                "sign_id": sign_id,
                "confidence": conf,
            })

        final_sign_id: Optional[str] = None
        final_confidence: float = 0.0

        if len(single_hand_predictions) == 2:
            s1, _ = single_hand_predictions[0]
            s2, _ = single_hand_predictions[1]
            pair = {s1, s2}
            if pair.intersection({"hello", "thank_you", "stop"}) and pair.intersection({"sorry", "yes_good"}):
                final_sign_id = "help"
                final_confidence = 0.97

        if final_sign_id is None and single_hand_predictions:
            single_hand_predictions.sort(key=lambda x: x[1], reverse=True)
            final_sign_id, final_confidence = single_hand_predictions[0]

        detected_sign_meta = None
        if final_sign_id:
            if final_sign_id in SIGN_DICTIONARY:
                detected_sign_meta = {
                    **SIGN_DICTIONARY[final_sign_id],
                    "confidence": final_confidence,
                    "is_custom": False,
                }
            else:
                for csign in self.custom_signs:
                    if csign["id"] == final_sign_id:
                        detected_sign_meta = {
                            k: v for k, v in csign.items() if k != "feature_vector"
                        }
                        detected_sign_meta["confidence"] = final_confidence
                        break

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        return {
            "hands": hands_output,
            "connections": HAND_CONNECTIONS,
            "detected_sign": detected_sign_meta,
            "latency_ms": latency_ms,
        }

