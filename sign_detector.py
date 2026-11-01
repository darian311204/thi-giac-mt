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

# Built-in Sign Language Dictionary (36 Common Sign Language Gestures)
SIGN_DICTIONARY: Dict[str, Dict[str, Any]] = {
    # --- 1. Nhóm Chào hỏi & Lịch sự (5) ---
    "hello": {
        "id": "hello",
        "label_vi": "Xin chào",
        "label_en": "Hello",
        "emoji": "👋",
        "category": "Chào hỏi",
        "how_to": "Giơ bàn tay mở rộng, xòe cả 5 ngón tay hướng lên trên.",
        "color": "#3b82f6"
    },
    "thank_you": {
        "id": "thank_you",
        "label_vi": "Cảm ơn",
        "label_en": "Thank You",
        "emoji": "🙏",
        "category": "Chào hỏi",
        "how_to": "Duỗi thẳng 5 ngón tay và khép sát 4 ngón chính vào nhau.",
        "color": "#10b981"
    },
    "peace": {
        "id": "peace",
        "label_vi": "Tạm biệt / Hòa bình",
        "label_en": "Peace / Goodbye",
        "emoji": "✌️",
        "category": "Chào hỏi",
        "how_to": "Giơ ngón trỏ và ngón giữa tách hình chữ V, các ngón còn lại gập xuống.",
        "color": "#8b5cf6"
    },
    "sorry": {
        "id": "sorry",
        "label_vi": "Xin lỗi",
        "label_en": "Sorry",
        "emoji": "✊",
        "category": "Chào hỏi",
        "how_to": "Nắm gọn cả 5 ngón tay thành nắm đấm trước ngực.",
        "color": "#f59e0b"
    },
    "please": {
        "id": "please",
        "label_vi": "Làm ơn / Xin vui lòng",
        "label_en": "Please",
        "emoji": "🫳",
        "category": "Chào hỏi",
        "how_to": "Mở cả 5 ngón tay và đặt bàn tay nằm ngang trước ngực.",
        "color": "#06b6d4"
    },

    # --- 2. Nhóm Phản hồi & Xác nhận (6) ---
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
        "how_to": "Chạm đầu ngón cái và đầu ngón trỏ thành vòng tròn, duỗi thẳng 3 ngón còn lại.",
        "color": "#0ea5e9"
    },
    "correct": {
        "id": "correct",
        "label_vi": "Đúng rồi / Chính xác",
        "label_en": "Correct / Right",
        "emoji": "✅",
        "category": "Phản hồi",
        "how_to": "Duỗi ngón cái và ngón trỏ tạo góc chữ L, gập 3 ngón giữa, áp út và út.",
        "color": "#10b981"
    },
    "wrong": {
        "id": "wrong",
        "label_vi": "Sai rồi / Không phải",
        "label_en": "Wrong / False",
        "emoji": "❌",
        "category": "Phản hồi",
        "how_to": "Chỉ duỗi thẳng ngón út hướng lên, gập 4 ngón còn lại.",
        "color": "#f43f5e"
    },
    "maybe": {
        "id": "maybe",
        "label_vi": "Có thể / Chưa chắc",
        "label_en": "Maybe / Unsure",
        "emoji": "🤔",
        "category": "Phản hồi",
        "how_to": "Duỗi ngón trỏ và ngón áp út, gập ngón cái, ngón giữa và ngón út.",
        "color": "#a855f7"
    },

    # --- 3. Nhóm Giao tiếp Hằng ngày (7) ---
    "point_one": {
        "id": "point_one",
        "label_vi": "Tôi / Chú ý",
        "label_en": "Me / Attention",
        "emoji": "☝️",
        "category": "Giao tiếp",
        "how_to": "Chỉ duỗi thẳng ngón trỏ hướng lên trên, gập 4 ngón còn lại.",
        "color": "#6366f1"
    },
    "you": {
        "id": "you",
        "label_vi": "Bạn / Người đối diện",
        "label_en": "You",
        "emoji": "👉",
        "category": "Giao tiếp",
        "how_to": "Duỗi ngón trỏ hướng ngang về phía trước, gập 4 ngón còn lại.",
        "color": "#3b82f6"
    },
    "call_me": {
        "id": "call_me",
        "label_vi": "Gọi cho tôi",
        "label_en": "Call Me",
        "emoji": "🤙",
        "category": "Giao tiếp",
        "how_to": "Duỗi ngón cái và ngón út ra ngoài, gập 3 ngón giữa vào lòng bàn tay.",
        "color": "#14b8a6"
    },
    "question": {
        "id": "question",
        "label_vi": "Câu hỏi / Tại sao?",
        "label_en": "Question / Why",
        "emoji": "❓",
        "category": "Giao tiếp",
        "how_to": "Gập ngón cái và ngón trỏ, duỗi thẳng 3 ngón giữa, áp út và út.",
        "color": "#f97316"
    },
    "understand": {
        "id": "understand",
        "label_vi": "Hiểu rồi",
        "label_en": "Understand",
        "emoji": "💡",
        "category": "Giao tiếp",
        "how_to": "Nắm 4 ngón tay và duỗi ngón cái hướng ngang (tư thế chữ A).",
        "color": "#eab308"
    },
    "dont_understand": {
        "id": "dont_understand",
        "label_vi": "Không hiểu",
        "label_en": "Don't Understand",
        "emoji": "🤷",
        "category": "Giao tiếp",
        "how_to": "Chỉ duỗi thẳng ngón giữa hướng lên, gập ngón cái, trỏ, áp út và út.",
        "color": "#64748b"
    },
    "name": {
        "id": "name",
        "label_vi": "Tên / Giới thiệu",
        "label_en": "Name",
        "emoji": "🏷️",
        "category": "Giao tiếp",
        "how_to": "Duỗi thẳng ngón trỏ và ngón giữa khép sát vào nhau (ký hiệu chữ U/H).",
        "color": "#06b6d4"
    },

    # --- 4. Nhóm Cảm xúc & Tinh thần (5) ---
    "i_love_you": {
        "id": "i_love_you",
        "label_vi": "Tôi yêu bạn",
        "label_en": "I Love You",
        "emoji": "🤟",
        "category": "Cảm xúc",
        "how_to": "Duỗi thẳng ngón cái, ngón trỏ và ngón út; gập ngón giữa và ngón áp út.",
        "color": "#ec4899"
    },
    "happy": {
        "id": "happy",
        "label_vi": "Vui vẻ / Hạnh phúc",
        "label_en": "Happy",
        "emoji": "😊",
        "category": "Cảm xúc",
        "how_to": "Duỗi ngón cái, ngón trỏ, ngón giữa và ngón út; chỉ gập ngón áp út.",
        "color": "#22c55e"
    },
    "sad": {
        "id": "sad",
        "label_vi": "Buồn bã",
        "label_en": "Sad",
        "emoji": "😢",
        "category": "Cảm xúc",
        "how_to": "Duỗi 4 ngón chính hướng chúc xuống dưới, gập ngón cái.",
        "color": "#60a5fa"
    },
    "wish_luck": {
        "id": "wish_luck",
        "label_vi": "Chúc may mắn / Hy vọng",
        "label_en": "Good Luck / Hope",
        "emoji": "🤞",
        "category": "Cảm xúc",
        "how_to": "Duỗi ngón trỏ và ngón giữa bắt chéo sát vào nhau (ký hiệu chữ R).",
        "color": "#10b981"
    },
    "awesome": {
        "id": "awesome",
        "label_vi": "Tuyệt vời / Mạnh mẽ",
        "label_en": "Awesome / Rock",
        "emoji": "🤘",
        "category": "Cảm xúc",
        "how_to": "Duỗi thẳng ngón trỏ và ngón út, gập ngón cái, ngón giữa và ngón áp út.",
        "color": "#f43f5e"
    },

    # --- 5. Nhóm Nhu cầu & Sinh hoạt (6) ---
    "eat": {
        "id": "eat",
        "label_vi": "Ăn cơm / Đói bụng",
        "label_en": "Eat / Hungry",
        "emoji": "🍚",
        "category": "Sinh hoạt",
        "how_to": "Chụm 5 đầu ngón tay chạm sát vào nhau tạo hình búp tay.",
        "color": "#f59e0b"
    },
    "drink": {
        "id": "drink",
        "label_vi": "Uống nước / Khát",
        "label_en": "Drink / Water",
        "emoji": "🥤",
        "category": "Sinh hoạt",
        "how_to": "Uốn cong 4 ngón tay và mở ngón cái tạo thành hình chữ C như cầm cốc.",
        "color": "#38bdf8"
    },
    "money": {
        "id": "money",
        "label_vi": "Tiền / Thanh toán",
        "label_en": "Money / Pay",
        "emoji": "💰",
        "category": "Sinh hoạt",
        "how_to": "Duỗi ngón cái và ngón giữa, gập ngón trỏ, áp út và út.",
        "color": "#eab308"
    },
    "home": {
        "id": "home",
        "label_vi": "Nhà / Về nhà",
        "label_en": "Home",
        "emoji": "🏠",
        "category": "Sinh hoạt",
        "how_to": "Duỗi ngón cái, ngón giữa và ngón áp út; gập ngón trỏ và ngón út.",
        "color": "#8b5cf6"
    },
    "restroom": {
        "id": "restroom",
        "label_vi": "Nhà vệ sinh",
        "label_en": "Restroom / Toilet",
        "emoji": "🚻",
        "category": "Sinh hoạt",
        "how_to": "Chỉ duỗi thẳng ngón áp út, gập 4 ngón còn lại.",
        "color": "#06b6d4"
    },
    "sleep": {
        "id": "sleep",
        "label_vi": "Đi ngủ / Nghỉ ngơi",
        "label_en": "Sleep / Rest",
        "emoji": "😴",
        "category": "Sinh hoạt",
        "how_to": "Duỗi ngón cái, ngón áp út và ngón út; gập ngón trỏ và ngón giữa.",
        "color": "#6366f1"
    },

    # --- 6. Nhóm Khẩn cấp & Y tế (4) ---
    "help": {
        "id": "help",
        "label_vi": "Cần giúp đỡ",
        "label_en": "Help Me",
        "emoji": "🆘",
        "category": "Khẩn cấp",
        "how_to": "Giơ 3 ngón giữa (trỏ, giữa, áp út) thẳng lên hoặc kết hợp 2 bàn tay.",
        "color": "#ef4444"
    },
    "stop": {
        "id": "stop",
        "label_vi": "Dừng lại / Khoan đã",
        "label_en": "Stop / Wait",
        "emoji": "✋",
        "category": "Khẩn cấp",
        "how_to": "Duỗi thẳng 4 ngón tay hướng lên và gập ngón cái vào trong lòng bàn tay.",
        "color": "#eab308"
    },
    "doctor": {
        "id": "doctor",
        "label_vi": "Bác sĩ / Bệnh viện",
        "label_en": "Doctor / Hospital",
        "emoji": "🏥",
        "category": "Khẩn cấp",
        "how_to": "Duỗi thẳng ngón giữa và ngón áp út, gập ngón cái, ngón trỏ và ngón út.",
        "color": "#10b981"
    },
    "pain": {
        "id": "pain",
        "label_vi": "Bị đau / Bị thương",
        "label_en": "Pain / Hurt",
        "emoji": "🤕",
        "category": "Khẩn cấp",
        "how_to": "Duỗi ngón cái và ngón áp út, gập ngón trỏ, ngón giữa và ngón út.",
        "color": "#f43f5e"
    },

    # --- 7. Nhóm Số đếm & Ký hiệu Bổ sung (3) ---
    "number_three": {
        "id": "number_three",
        "label_vi": "Số 3 / Ba người",
        "label_en": "Number Three",
        "emoji": "3️⃣",
        "category": "Số đếm",
        "how_to": "Duỗi thẳng ngón cái, ngón trỏ và ngón giữa; gập ngón áp út và ngón út.",
        "color": "#3b82f6"
    },
    "number_six": {
        "id": "number_six",
        "label_vi": "Số 6 / Sáu",
        "label_en": "Number Six",
        "emoji": "6️⃣",
        "category": "Số đếm",
        "how_to": "Chạm đầu ngón cái vào đầu ngón út, duỗi thẳng 3 ngón trỏ, giữa, áp út.",
        "color": "#8b5cf6"
    },
    "number_seven": {
        "id": "number_seven",
        "label_vi": "Số 7 / Bảy",
        "label_en": "Number Seven",
        "emoji": "7️⃣",
        "category": "Số đếm",
        "how_to": "Duỗi thẳng ngón trỏ, ngón giữa và ngón út; gập ngón áp út chạm ngón cái.",
        "color": "#14b8a6"
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


def _angle_deg(a: Dict[str, float], b: Dict[str, float], c: Dict[str, float]) -> float:
    """Compute angle ABC in degrees in 3D."""
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
        """Determine extension state and geometric properties of all 5 fingers."""
        wrist = lm[0]
        palm_width = _dist2d(lm[5], lm[17]) + 1e-6
        palm_length = _dist2d(lm[0], lm[9]) + 1e-6
        scale = (palm_width + palm_length) * 0.5

        def is_finger_extended(mcp_idx: int, pip_idx: int, tip_idx: int) -> Tuple[bool, float]:
            dist_tip_wrist = _dist2d(lm[tip_idx], wrist)
            dist_pip_wrist = _dist2d(lm[pip_idx], wrist)
            dist_mcp_wrist = _dist2d(lm[mcp_idx], wrist)
            pip_angle = _angle_deg(lm[mcp_idx], lm[pip_idx], lm[tip_idx])
            ext = (
                dist_tip_wrist > dist_pip_wrist * 1.08
                and dist_tip_wrist > dist_mcp_wrist * 1.25
                and pip_angle > 130.0
            )
            return ext, pip_angle

        index_up, index_angle = is_finger_extended(5, 6, 8)
        middle_up, middle_angle = is_finger_extended(9, 10, 12)
        ring_up, ring_angle = is_finger_extended(13, 14, 16)
        pinky_up, pinky_angle = is_finger_extended(17, 18, 20)

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
            and dist_thumb_middle_pip > 0.48
        )

        return {
            "thumb": thumb_up,
            "index": index_up,
            "middle": middle_up,
            "ring": ring_up,
            "pinky": pinky_up,
            "pip_angles": (index_angle, middle_angle, ring_angle, pinky_angle),
            "scale": scale,
            "palm_width": palm_width,
            "palm_length": palm_length,
        }

    def classify_single_hand(
        self, lm: List[Dict[str, float]], finger_info: Dict[str, Any]
    ) -> Tuple[Optional[str], float]:
        """Classify a single hand's 21 landmarks into one of 36 built-in signs or custom signs."""
        # 1. Check custom user-trained signs first
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
        pw = finger_info["palm_width"]
        pip_angles = finger_info["pip_angles"]

        # Hand orientation checks
        dx_palm = lm[9]["x"] - lm[0]["x"]
        dy_palm = lm[9]["y"] - lm[0]["y"]
        hand_sideways = abs(dx_palm) > abs(dy_palm) * 1.15
        hand_down = dy_palm > 0.08 * pw

        # Pinch distances normalized by palm width
        pinch_index = _dist2d(lm[4], lm[8]) / pw
        pinch_pinky = _dist2d(lm[4], lm[20]) / pw

        # Special Check A: OK sign (#8) -> Thumb & Index pinched, Middle + Ring + Pinky extended
        if pinch_index < 0.40 and middle and ring and pinky and (not index):
            return "ok", 0.98

        # Special Check B: Number Six (#35) -> Thumb & Pinky pinched, Index + Middle + Ring extended
        if pinch_pinky < 0.40 and index and middle and ring and (not pinky):
            return "number_six", 0.95

        # Special Check C: Eat / Bunched O-hand (#24) -> All 4 fingertips bunched close to thumb tip
        if (
            (not thumb)
            and (not index)
            and (not middle)
            and (not ring)
            and (not pinky)
            and _dist2d(lm[4], lm[8]) / pw < 0.38
            and _dist2d(lm[8], lm[12]) / pw < 0.35
            and _dist2d(lm[12], lm[16]) / pw < 0.35
        ):
            return "eat", 0.94

        # Special Check D: Drink / C-cup hand (#25) -> 4 fingers half-curved (angles 92..126) & thumb open
        if (
            thumb
            and (not index)
            and (not middle)
            and (not ring)
            and (not pinky)
            and all(90.0 <= ang <= 126.0 for ang in pip_angles)
        ):
            return "drink", 0.93

        # Case 1: All 4 main fingers extended (Index, Middle, Ring, Pinky)
        if index and middle and ring and pinky:
            spread = (
                _dist2d(lm[8], lm[12])
                + _dist2d(lm[12], lm[16])
                + _dist2d(lm[16], lm[20])
            ) / pw

            if not thumb:
                if hand_down:
                    return "sad", 0.92
                return "stop", 0.93

            if hand_sideways:
                return "please", 0.94
            if spread >= 0.98:
                return "hello", 0.98
            return "thank_you", 0.94

        # Case 2: Index + Middle extended, Ring + Pinky curled (T, 1, 1, 0, 0)
        if index and middle and (not ring) and (not pinky):
            if thumb:
                return "number_three", 0.95
            dist_im = _dist2d(lm[8], lm[12]) / pw
            # Crossed index & middle (R / Wish luck)
            if lm[8]["x"] > lm[12]["x"] + 0.01 * pw:
                return "wish_luck", 0.93
            # Tight parallel index & middle (U / Name)
            if dist_im < 0.30:
                return "name", 0.94
            # Spread V-sign (Peace / Goodbye)
            return "peace", 0.96

        # Case 3: Only Index extended (T, 1, 0, 0, 0)
        if index and (not middle) and (not ring) and (not pinky):
            if thumb:
                return "correct", 0.95
            dx_idx = abs(lm[8]["x"] - lm[5]["x"])
            dy_idx = abs(lm[8]["y"] - lm[5]["y"])
            if hand_sideways or dx_idx > dy_idx * 1.15:
                return "you", 0.93
            return "point_one", 0.94

        # Case 4: All 4 main fingers curled (T, 0, 0, 0, 0)
        if (not index) and (not middle) and (not ring) and (not pinky):
            dy_wrist = lm[4]["y"] - lm[0]["y"]
            dy_mcp = lm[4]["y"] - lm[5]["y"]
            thumb_reach = _dist2d(lm[4], lm[9]) / pw

            if thumb or thumb_reach > 0.65:
                if dy_mcp < -0.75 * pw and dy_wrist < -0.20 * pw:
                    return "yes_good", 0.96
                if dy_mcp > 0.25 * pw and dy_wrist > 0.15 * pw:
                    return "no_bad", 0.95
                return "understand", 0.92
            return "sorry", 0.91

        # Case 5: Remaining 1-to-1 Finger Bitmasks (T, I, M, R, P)
        state_tuple = (thumb, index, middle, ring, pinky)
        bitmask_map: Dict[Tuple[bool, bool, bool, bool, bool], Tuple[str, float]] = {
            (True, True, False, False, True): ("i_love_you", 0.97),
            (True, False, False, False, True): ("call_me", 0.95),
            (False, True, True, True, False): ("help", 0.93),
            (False, False, False, False, True): ("wrong", 0.93),
            (False, True, False, True, False): ("maybe", 0.91),
            (False, False, True, True, True): ("question", 0.92),
            (False, False, True, False, False): ("dont_understand", 0.92),
            (True, True, True, False, True): ("happy", 0.94),
            (False, True, False, False, True): ("awesome", 0.95),
            (True, False, True, False, False): ("money", 0.92),
            (True, False, True, True, False): ("home", 0.92),
            (False, False, False, True, False): ("restroom", 0.91),
            (True, False, False, True, True): ("sleep", 0.93),
            (False, False, True, True, False): ("doctor", 0.92),
            (True, False, False, True, False): ("pain", 0.91),
            (False, True, True, False, True): ("number_seven", 0.94),
        }

        if state_tuple in bitmask_map:
            return bitmask_map[state_tuple]

        return None, 0.0

    def process_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
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

    def process_landmarks_payload(self, hands_input: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process 21-point hand landmarks through the Python geometric classifier."""
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
