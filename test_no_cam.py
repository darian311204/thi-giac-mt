"""
Kịch bản kiểm thử Nhận diện Ngôn ngữ Ký hiệu KHÔNG CẦN CAMERA (No-Cam Test Suite)
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

    if thumb_down:
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

    def chain(idx: int, up: bool, fan: float, ok_idx: bool = False) -> List[Dict[str, float]]:
        bx, by = mcp_x[idx], mcp_y
        if ok_idx:
            return [
                {"x": bx, "y": by, "z": 0.0},
                {"x": bx - 0.015, "y": by - 0.08, "z": 0.0},
                {"x": bx + 0.005, "y": by - 0.095, "z": 0.0},
                {"x": 0.428, "y": 0.482, "z": 0.0},
            ]
        if up:
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

    return (
        [wrist]
        + thumb
        + chain(0, fingers["index"], -1.5, ok)
        + chain(1, fingers["middle"], -0.5)
        + chain(2, fingers["ring"], 0.5)
        + chain(3, fingers["pinky"], 1.5)
    )


def render_screenshot(lms: List[Dict[str, float]], det_sign: Dict[str, Any], filename: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    W, H = 960, 600
    img = Image.new("RGB", (W, H), (7, 11, 20))
    draw = ImageDraw.Draw(img)

    # Cyber grid background
    for x in range(0, W, 32):
        draw.line([(x, 0), (x, H)], fill=(18, 30, 56), width=1)
    for y in range(0, H, 32):
        draw.line([(0, y), (W, y)], fill=(18, 30, 56), width=1)

    pts = [(int(p["x"] * W), int(p["y"] * H)) for p in lms]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    draw.rectangle(
        [min(xs) - 28, min(ys) - 28, max(xs) + 28, max(ys) + 28],
        outline=(59, 130, 246),
        width=2,
    )

    for i, j in HAND_CONNECTIONS:
        draw.line([pts[i], pts[j]], fill=(56, 189, 248), width=5)

    for idx, (cx, cy) in enumerate(pts):
        r = 8 if idx in (4, 8, 12, 16, 20) else 5
        color = (16, 185, 129) if idx in (4, 8, 12, 16, 20) else (255, 255, 255)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(15, 23, 42), width=2)

    # Bottom HUD bar
    draw.rectangle([0, H - 92, W, H], fill=(15, 23, 42))
    draw.line([(0, H - 92), (W, H - 92)], fill=(59, 130, 246), width=3)

    try:
        font_lg = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 24)
        font_sm = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except Exception:
        font_lg = ImageFont.load_default()
        font_sm = ImageFont.load_default()

    title = f"SignVision AI  |  {det_sign['label_vi']} ({det_sign['label_en']})"
    subtitle = (
        f"Độ tin cậy: {int(det_sign['confidence'] * 100)}%   |   "
        f"Nhóm: {det_sign['category']}   |   {det_sign['how_to']}"
    )
    draw.text((24, H - 74), title, fill=(255, 255, 255), font=font_lg)
    draw.text((24, H - 36), subtitle, fill=(147, 197, 253), font=font_sm)

    out_path = os.path.join(OUT_DIR, filename)
    img.save(out_path)
    return out_path


def run_tests() -> None:
    detector = SignLanguageDetector()
    test_cases = [
        ("hello", {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}, {}),
        ("thank_you", {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}, {"tight": True}),
        ("i_love_you", {"thumb": True, "index": True, "middle": False, "ring": False, "pinky": True}, {}),
        ("ok", {"thumb": True, "index": False, "middle": True, "ring": True, "pinky": True}, {"ok": True}),
        ("yes_good", {"thumb": True, "index": False, "middle": False, "ring": False, "pinky": False}, {"thumb_up": True}),
        ("no_bad", {"thumb": True, "index": False, "middle": False, "ring": False, "pinky": False}, {"thumb_down": True}),
        ("sorry", {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}, {}),
        ("peace", {"thumb": False, "index": True, "middle": True, "ring": False, "pinky": False}, {}),
        ("call_me", {"thumb": True, "index": False, "middle": False, "ring": False, "pinky": True}, {}),
        ("point_one", {"thumb": False, "index": True, "middle": False, "ring": False, "pinky": False}, {}),
        ("help", {"thumb": False, "index": True, "middle": True, "ring": True, "pinky": False}, {}),
        ("stop", {"thumb": False, "index": True, "middle": True, "ring": True, "pinky": True}, {}),
    ]

    print("=" * 72)
    print("KIỂM THỬ NHẬN DIỆN 12 KÝ HIỆU KHÔNG CẦN CAMERA (NO-CAM TEST SUITE)")
    print("=" * 72)
    passed = 0
    for expected_id, f_state, opts in test_cases:
        lms = build_pose(f_state, **opts)
        res = detector.process_landmarks_payload([{"handedness": "Right", "score": 0.98, "landmarks": lms}])
        det = res["detected_sign"]
        actual_id = det["id"] if det else None
        ok_match = actual_id == expected_id
        if ok_match:
            passed += 1
            shot_path = render_screenshot(lms, det, f"screenshot_{expected_id}.png")
            status = "PASS"
        else:
            shot_path = ""
            status = "FAIL"
        label_str = f"{det['label_vi']} ({det['label_en']})" if det else "None"
        conf_str = f"{int(det['confidence'] * 100)}%" if det else "0%"
        print(f"[{status}] Kỳ vọng: {expected_id:11s} | Nhận diện: {str(actual_id):11s} | {label_str:32s} | Tin cậy: {conf_str}")

    print("-" * 72)
    print(f"Kết quả: Đạt {passed}/{len(test_cases)} ký hiệu. Đã lưu {passed} ảnh vào thư mục screenshots/")


if __name__ == "__main__":
    run_tests()
