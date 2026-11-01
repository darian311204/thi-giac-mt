"""
Kịch bản kiểm thử Nhận diện 36 Ngôn ngữ Ký hiệu KHÔNG CẦN CAMERA (No-Cam Test Suite)
và tự động xuất ảnh chụp màn hình (Screenshot) kết quả nhận diện vào thư mục screenshots/.
"""
import os
import sys
from typing import Dict, List, Any
from PIL import Image, ImageDraw, ImageFont
from sign_detector import SignLanguageDetector, HAND_CONNECTIONS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")


def build_pose(
    fingers: Dict[str, bool],
    tight: bool = False,
    ok: bool = False,
    pinch_pinky: bool = False,
    o_bunched: bool = False,
    c_cup: bool = False,
    crossed_im: bool = False,
    tight_im: bool = False,
    hand_sideways: bool = False,
    hand_down: bool = False,
    thumb_up: bool = False,
    thumb_down: bool = False,
) -> List[Dict[str, float]]:
    knuckle_gap = 0.056
    wrist = {"x": 0.5, "y": 0.78, "z": 0.0}
    mcp_x = [
        0.5 - 1.5 * knuckle_gap,
        0.5 - 0.5 * knuckle_gap,
        0.5 + 0.5 * knuckle_gap,
        0.5 + 1.5 * knuckle_gap,
    ]
    mcp_y = 0.56

    if o_bunched:
        thumb = [
            {"x": 0.44, "y": 0.68, "z": 0.0},
            {"x": 0.42, "y": 0.60, "z": 0.0},
            {"x": 0.44, "y": 0.53, "z": 0.0},
            {"x": 0.48, "y": 0.49, "z": 0.0},
        ]
    elif pinch_pinky:
        thumb = [
            {"x": 0.44, "y": 0.68, "z": 0.0},
            {"x": 0.48, "y": 0.63, "z": 0.0},
            {"x": 0.53, "y": 0.59, "z": 0.0},
            {"x": 0.575, "y": 0.55, "z": 0.0},
        ]
    elif thumb_down:
        thumb = [
            {"x": 0.43, "y": 0.66, "z": 0.0},
            {"x": 0.39, "y": 0.73, "z": 0.0},
            {"x": 0.37, "y": 0.81, "z": 0.0},
            {"x": 0.35, "y": 0.89, "z": 0.0},
        ]
    elif thumb_up:
        thumb = [
            {"x": 0.44, "y": 0.66, "z": 0.0},
            {"x": 0.40, "y": 0.56, "z": 0.0},
            {"x": 0.38, "y": 0.46, "z": 0.0},
            {"x": 0.36, "y": 0.36, "z": 0.0},
        ]
    elif ok:
        thumb = [
            {"x": 0.44, "y": 0.68, "z": 0.0},
            {"x": 0.40, "y": 0.60, "z": 0.0},
            {"x": 0.40, "y": 0.53, "z": 0.0},
            {"x": 0.425, "y": 0.48, "z": 0.0},
        ]
    elif fingers["thumb"]:
        thumb = [
            {"x": 0.44, "y": 0.69, "z": 0.0},
            {"x": 0.38, "y": 0.62, "z": 0.0},
            {"x": 0.32, "y": 0.55, "z": 0.0},
            {"x": 0.26, "y": 0.48, "z": 0.0},
        ]
    else:
        thumb = [
            {"x": 0.44, "y": 0.69, "z": 0.0},
            {"x": 0.42, "y": 0.62, "z": 0.0},
            {"x": 0.46, "y": 0.60, "z": 0.0},
            {"x": 0.49, "y": 0.61, "z": 0.0},
        ]

    def chain(idx: int, up: bool, fan: float) -> List[Dict[str, float]]:
        bx, by = mcp_x[idx], mcp_y
        if o_bunched:
            # All fingertips converge around (0.485, 0.49)
            tx = 0.475 + idx * 0.008
            ty = 0.49
            return [
                {"x": bx, "y": by, "z": 0.0},
                {"x": bx, "y": by - 0.06, "z": 0.0},
                {"x": (bx + tx) * 0.5, "y": by - 0.08, "z": 0.0},
                {"x": tx, "y": ty, "z": 0.0},
            ]
        if c_cup:
            # Half-curved C-cup fingers: angle(MCP, PIP, TIP) ~ 110 degrees
            return [
                {"x": bx, "y": by, "z": 0.0},
                {"x": bx - 0.02, "y": by - 0.10, "z": 0.0},
                {"x": bx - 0.07, "y": by - 0.12, "z": 0.0},
                {"x": bx - 0.11, "y": by - 0.09, "z": 0.0},
            ]
        if ok and idx == 0:
            return [
                {"x": bx, "y": by, "z": 0.0},
                {"x": bx - 0.015, "y": by - 0.08, "z": 0.0},
                {"x": bx + 0.005, "y": by - 0.095, "z": 0.0},
                {"x": 0.428, "y": 0.482, "z": 0.0},
            ]
        if pinch_pinky and idx == 3:
            return [
                {"x": bx, "y": by, "z": 0.0},
                {"x": bx, "y": by - 0.06, "z": 0.0},
                {"x": 0.58, "y": 0.53, "z": 0.0},
                {"x": 0.578, "y": 0.552, "z": 0.0},
            ]
        if up:
            if crossed_im and idx in (0, 1):
                # Cross index tip (idx 0) to the right of middle tip (idx 1)
                target_x = 0.488 if idx == 0 else 0.462
                return [
                    {"x": bx, "y": by, "z": 0.0},
                    {"x": (bx + target_x) * 0.5, "y": by - 0.095, "z": 0.0},
                    {"x": target_x, "y": by - 0.185, "z": 0.0},
                    {"x": target_x, "y": by - 0.275, "z": 0.0},
                ]
            if tight_im and idx in (0, 1):
                target_x = 0.465 if idx == 0 else 0.482
                return [
                    {"x": bx, "y": by, "z": 0.0},
                    {"x": target_x, "y": by - 0.095, "z": 0.0},
                    {"x": target_x, "y": by - 0.185, "z": 0.0},
                    {"x": target_x, "y": by - 0.275, "z": 0.0},
                ]
            ts = -fan * 0.018 if tight else fan * 0.028
            return [
                {"x": bx, "y": by, "z": 0.0},
                {"x": bx + ts * 0.33, "y": by - 0.095, "z": 0.0},
                {"x": bx + ts * 0.66, "y": by - 0.185, "z": 0.0},
                {"x": bx + ts, "y": by - 0.275, "z": 0.0},
            ]
        return [
            {"x": bx, "y": by, "z": 0.0},
            {"x": bx, "y": by - 0.055, "z": 0.0},
            {"x": bx, "y": by + 0.015, "z": 0.0},
            {"x": bx, "y": by + 0.045, "z": 0.0},
        ]

    pts = (
        [wrist]
        + thumb
        + chain(0, fingers["index"], -1.5)
        + chain(1, fingers["middle"], -0.5)
        + chain(2, fingers["ring"], 0.5)
        + chain(3, fingers["pinky"], 1.5)
    )

    if hand_sideways:
        # Rotate 90 degrees around center (0.5, 0.55)
        cx, cy = 0.5, 0.55
        pts = [
            {"x": round(cx + (p["y"] - cy), 4), "y": round(cy - (p["x"] - cx), 4), "z": 0.0}
            for p in pts
        ]
    elif hand_down:
        # Flip vertically around center y = 0.52
        cy = 0.52
        pts = [
            {"x": p["x"], "y": round(2 * cy - p["y"], 4), "z": 0.0}
            for p in pts
        ]

    return pts


TEST_CASES = [
    # 1. Chào hỏi (5)
    ("hello", {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}, {}),
    ("thank_you", {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}, {"tight": True}),
    ("peace", {"thumb": False, "index": True, "middle": True, "ring": False, "pinky": False}, {}),
    ("sorry", {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}, {}),
    ("please", {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}, {"hand_sideways": True}),
    # 2. Phản hồi (6)
    ("yes_good", {"thumb": True, "index": False, "middle": False, "ring": False, "pinky": False}, {"thumb_up": True}),
    ("no_bad", {"thumb": True, "index": False, "middle": False, "ring": False, "pinky": False}, {"thumb_down": True}),
    ("ok", {"thumb": True, "index": False, "middle": True, "ring": True, "pinky": True}, {"ok": True}),
    ("correct", {"thumb": True, "index": True, "middle": False, "ring": False, "pinky": False}, {}),
    ("wrong", {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": True}, {}),
    ("maybe", {"thumb": False, "index": True, "middle": False, "ring": True, "pinky": False}, {}),
    # 3. Giao tiếp (7)
    ("point_one", {"thumb": False, "index": True, "middle": False, "ring": False, "pinky": False}, {}),
    ("you", {"thumb": False, "index": True, "middle": False, "ring": False, "pinky": False}, {"hand_sideways": True}),
    ("call_me", {"thumb": True, "index": False, "middle": False, "ring": False, "pinky": True}, {}),
    ("question", {"thumb": False, "index": False, "middle": True, "ring": True, "pinky": True}, {}),
    ("understand", {"thumb": True, "index": False, "middle": False, "ring": False, "pinky": False}, {}),
    ("dont_understand", {"thumb": False, "index": False, "middle": True, "ring": False, "pinky": False}, {}),
    ("name", {"thumb": False, "index": True, "middle": True, "ring": False, "pinky": False}, {"tight_im": True}),
    # 4. Cảm xúc (5)
    ("i_love_you", {"thumb": True, "index": True, "middle": False, "ring": False, "pinky": True}, {}),
    ("happy", {"thumb": True, "index": True, "middle": True, "ring": False, "pinky": True}, {}),
    ("sad", {"thumb": False, "index": True, "middle": True, "ring": True, "pinky": True}, {"hand_down": True}),
    ("wish_luck", {"thumb": False, "index": True, "middle": True, "ring": False, "pinky": False}, {"crossed_im": True}),
    ("awesome", {"thumb": False, "index": True, "middle": False, "ring": False, "pinky": True}, {}),
    # 5. Sinh hoạt (6)
    ("eat", {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}, {"o_bunched": True}),
    ("drink", {"thumb": True, "index": False, "middle": False, "ring": False, "pinky": False}, {"c_cup": True}),
    ("money", {"thumb": True, "index": False, "middle": True, "ring": False, "pinky": False}, {}),
    ("home", {"thumb": True, "index": False, "middle": True, "ring": True, "pinky": False}, {}),
    ("restroom", {"thumb": False, "index": False, "middle": False, "ring": True, "pinky": False}, {}),
    ("sleep", {"thumb": True, "index": False, "middle": False, "ring": True, "pinky": True}, {}),
    # 6. Khẩn cấp (4)
    ("help", {"thumb": False, "index": True, "middle": True, "ring": True, "pinky": False}, {}),
    ("stop", {"thumb": False, "index": True, "middle": True, "ring": True, "pinky": True}, {}),
    ("doctor", {"thumb": False, "index": False, "middle": True, "ring": True, "pinky": False}, {}),
    ("pain", {"thumb": True, "index": False, "middle": False, "ring": True, "pinky": False}, {}),
    # 7. Số đếm (3)
    ("number_three", {"thumb": True, "index": True, "middle": True, "ring": False, "pinky": False}, {}),
    ("number_six", {"thumb": False, "index": True, "middle": True, "ring": True, "pinky": False}, {"pinch_pinky": True}),
    ("number_seven", {"thumb": False, "index": True, "middle": True, "ring": False, "pinky": True}, {}),
]


def run_tests() -> None:
    detector = SignLanguageDetector()
    print("=" * 80)
    print(f"KIỂM THỬ NHẬN DIỆN {len(TEST_CASES)} NGÔN NGỮ KÝ HIỆU KHÔNG CẦN CAMERA (NO-CAM TEST SUITE)")
    print("=" * 80)
    passed = 0
    for expected_id, f_state, opts in TEST_CASES:
        lms = build_pose(f_state, **opts)
        res = detector.process_landmarks_payload([{"handedness": "Right", "score": 0.98, "landmarks": lms}])
        det = res["detected_sign"]
        actual_id = det["id"] if det else None
        ok_match = actual_id == expected_id
        if ok_match:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"
        label_str = f"{det['label_vi']} ({det['label_en']})" if det else "None"
        conf_str = f"{int(det['confidence'] * 100)}%" if det else "0%"
        print(f"[{status}] Kỳ vọng: {expected_id:16s} | Nhận diện: {str(actual_id):16s} | {label_str:36s} | {conf_str}")

    print("-" * 80)
    print(f"Kết quả tổng hợp: Đạt {passed}/{len(TEST_CASES)} ký hiệu ({round(passed * 100 / len(TEST_CASES), 1)}%).")


if __name__ == "__main__":
    run_tests()
