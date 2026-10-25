import os
import time
import base64
import json
import sqlite3
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from sign_detector import SignLanguageDetector, DB_FILE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

app = FastAPI(
    title="SignVision AI - Hệ thống Nhận diện Ngôn ngữ Ký hiệu",
    description="API Backend sử dụng OpenCV + MediaPipe HandLandmarker nhận diện ngôn ngữ ký hiệu thời gian thực",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/screenshots", StaticFiles(directory=SCREENSHOTS_DIR), name="screenshots")

detector = SignLanguageDetector()


class FrameRequest(BaseModel):
    image: str  # base64 data URL or raw base64 string


class LandmarksDetectRequest(BaseModel):
    hands: List[Dict[str, Any]]


class CustomSignRequest(BaseModel):
    id: str
    label_vi: str
    label_en: str
    emoji: Optional[str] = "🖐️"
    how_to: Optional[str] = "Ký hiệu tùy chỉnh do người dùng huấn luyện"
    landmarks: List[Dict[str, float]]


class HistoryCreateRequest(BaseModel):
    sign_id: str
    label_vi: str
    confidence: float


class ScreenshotSaveRequest(BaseModel):
    image: str
    sign_id: Optional[str] = "capture"
    label_vi: Optional[str] = "Ảnh chụp màn hình"


def decode_base64_image(data_str: str) -> bytes:
    if "," in data_str:
        data_str = data_str.split(",", 1)[1]
    return base64.b64decode(data_str)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "engine": "MediaPipe HandLandmarker + OpenCV",
        "total_signs": len(detector.get_all_signs()),
    }


@app.get("/api/signs")
def get_signs():
    return {"signs": detector.get_all_signs()}


@app.post("/api/detect")
def detect_from_frame(req: FrameRequest):
    try:
        img_bytes = decode_base64_image(req.image)
        result = detector.process_image_bytes(img_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý khung hình: {str(e)}")


@app.post("/api/detect-landmarks")
def detect_from_landmarks(req: LandmarksDetectRequest):
    try:
        return detector.process_landmarks_payload(req.hands)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi phân tích landmarks: {str(e)}")


@app.post("/api/screenshots")
def save_screenshot(req: ScreenshotSaveRequest):
    try:
        img_bytes = decode_base64_image(req.image)
        safe_tag = "".join(c if c.isalnum() or c == "_" else "_" for c in (req.sign_id or "capture"))
        filename = f"shot_{safe_tag}_{int(time.time() * 1000)}.png"
        filepath = os.path.join(SCREENSHOTS_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(img_bytes)
        return {
            "status": "ok",
            "filename": filename,
            "url": f"/screenshots/{filename}",
            "label_vi": req.label_vi or "Ảnh chụp màn hình",
            "created_at": time.strftime("%H:%M:%S %d/%m/%Y"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể lưu ảnh chụp màn hình: {str(e)}")


@app.get("/api/screenshots")
def list_screenshots():
    files = []
    if os.path.isdir(SCREENSHOTS_DIR):
        for fname in sorted(os.listdir(SCREENSHOTS_DIR), reverse=True):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                fpath = os.path.join(SCREENSHOTS_DIR, fname)
                mtime = os.path.getmtime(fpath)
                files.append({
                    "filename": fname,
                    "url": f"/screenshots/{fname}",
                    "created_at": time.strftime("%H:%M:%S %d/%m/%Y", time.localtime(mtime)),
                })
    return {"screenshots": files[:24]}


@app.delete("/api/screenshots/{filename}")
def delete_screenshot(filename: str):
    safe_name = os.path.basename(filename)
    fpath = os.path.join(SCREENSHOTS_DIR, safe_name)
    if os.path.isfile(fpath):
        os.remove(fpath)
    return {"status": "ok"}


@app.post("/api/custom-signs")
def create_custom_sign(req: CustomSignRequest):
    if len(req.landmarks) != 21:
        raise HTTPException(status_code=400, detail="Cần đủ 21 điểm mốc bàn tay (landmarks) để lưu ký hiệu.")
    saved = detector.save_custom_sign(
        sign_id=req.id.strip().lower().replace(" ", "_"),
        label_vi=req.label_vi.strip(),
        label_en=req.label_en.strip() or req.label_vi.strip(),
        emoji=req.emoji or "🖐️",
        how_to=req.how_to or "Ký hiệu tùy chỉnh",
        landmarks=req.landmarks,
    )
    return {"status": "ok", "sign": saved, "signs": detector.get_all_signs()}


@app.delete("/api/custom-signs/{sign_id}")
def remove_custom_sign(sign_id: str):
    deleted = detector.delete_custom_sign(sign_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Không tìm thấy ký hiệu tùy chỉnh.")
    return {"status": "ok", "signs": detector.get_all_signs()}


@app.get("/api/history")
def get_history(limit: int = 30):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, sign_id, label_vi, confidence, created_at FROM detection_history ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        return {
            "history": [
                {
                    "id": r[0],
                    "sign_id": r[1],
                    "label_vi": r[2],
                    "confidence": r[3],
                    "created_at": r[4],
                }
                for r in rows
            ]
        }


@app.post("/api/history")
def add_history(req: HistoryCreateRequest):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO detection_history (sign_id, label_vi, confidence) VALUES (?, ?, ?)",
            (req.sign_id, req.label_vi, req.confidence),
        )
        conn.commit()
    return {"status": "ok"}


@app.delete("/api/history")
def clear_history():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM detection_history")
        conn.commit()
    return {"status": "ok"}


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive()
            if "text" in message and message["text"]:
                payload = message["text"]
                try:
                    data = json.loads(payload)
                    if "landmarks_hands" in data:
                        result = detector.process_landmarks_payload(data["landmarks_hands"])
                        await websocket.send_json(result)
                        continue
                    img_b64 = data.get("image", "")
                except json.JSONDecodeError:
                    img_b64 = payload

                if not img_b64:
                    await websocket.send_json({"error": "Empty image", "hands": [], "detected_sign": None})
                    continue

                img_bytes = decode_base64_image(img_b64)
                result = detector.process_image_bytes(img_bytes)
                await websocket.send_json(result)
            elif "bytes" in message and message["bytes"]:
                result = detector.process_image_bytes(message["bytes"])
                await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


FRONTEND_DIST = os.path.join(BASE_DIR, "thigiacmt", "dist")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8008, reload=False)
