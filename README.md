# SignVision AI — Hệ thống Nhận diện Ngôn ngữ Ký hiệu qua Camera

Ứng dụng Web thời gian thực nhận diện **Ngôn ngữ Ký hiệu Giao tiếp Thông dụng** sử dụng **React + TypeScript (Vite)** kết nối với **Python Backend (FastAPI + OpenCV + MediaPipe HandLandmarker 21 3D Points)**.

Dự án tích hợp sẵn **Bộ Công cụ Kiểm thử Không Cần Camera (No-Cam Test Suite)** và tính năng **Chụp màn hình kết quả (Screenshot)** trực tiếp trên giao diện.

---

## ✨ Tính năng Chính

1. **Nhận diện 12 Ký hiệu Giao tiếp Thông dụng + Ký hiệu Tùy chỉnh**:
   - `👋 Xin chào`, `🙏 Cảm ơn`, `🤟 Tôi yêu bạn`, `👍 Đồng ý / Tốt`, `👎 Không đồng ý`, `👌 OK / Ổn`, `✊ Xin lỗi`, `✌️ Tạm biệt / Hòa bình`, `🤙 Gọi cho tôi`, `☝️ Tôi / Chú ý`, `🆘 Cần giúp đỡ`, `✋ Dừng lại`.
2. **Hỗ trợ Kiểm thử Toàn diện Khi Chưa Có Webcam (No-Cam Mode)**:
   - **🤖 Chạy Luồng Auto-Test**: Giả lập luồng khung xương bàn tay chuyển động tự nhiên gửi liên tục về Python Backend (`/api/detect-landmarks`) để kiểm thử tốc độ xử lý, ghép câu tự động và phát âm.
   - **📁 Tải Ảnh / Video hoặc Dán Ảnh (`Ctrl + V`)**: Tải file ảnh/video từ máy tính hoặc dán trực tiếp ảnh chụp màn hình (`Ctrl + V`) để MediaPipe phân tích.
   - **🖐️ Bàn tay Tương tác 5 Ngón**: Tự bấm gập/duỗi từng ngón tay (`Cái`, `Trỏ`, `Giữa`, `Áp út`, `Út`) để kiểm chứng thuật toán hình học trên Python Backend.
   - **🧪 Kịch bản Kiểm thử CLI (`test_no_cam.py`)**: Kiểm tra tự động 12/12 ký hiệu qua dòng lệnh.
3. **📸 Chụp Ảnh Màn hình Kết quả (Screenshot Gallery)**:
   - Chụp và ghép khung hình hiện tại + khung xương 21 điểm + hộp thông tin nhận diện + nhãn thời gian thành file `.png` lưu vào thư mục `screenshots/` và hiển thị ngay trên Web.
4. **💬 Ghép câu Giao tiếp & Phát âm (Text-to-Speech)**:
   - Tự động ghép các ký hiệu thành câu hoàn chỉnh và phát âm tiếng Việt (`vi-VN`) hoặc tiếng Anh (`en-US`).
5. **✨ Tự huấn luyện Ký hiệu mới (Custom Sign Trainer - SQLite)**:
   - Chụp mẫu vector 21 điểm mốc chuẩn hóa và lưu ký hiệu mới vào cơ sở dữ liệu `gestures.db`.

---

## 🛠️ Cài đặt trên Môi trường Mới Hoàn toàn

### Yêu cầu hệ thống cần cài trước (Prerequisites)
1. **Python 3.10 – 3.13**: Tải tại [python.org](https://www.python.org/downloads/) *(lưu ý tích chọn **"Add python.exe to PATH"** khi cài trên Windows)*.
2. **Node.js (v18 trở lên, khuyến nghị LTS v20/v22)**: Tải tại [nodejs.org](https://nodejs.org/) *(đi kèm `npm`)*.
3. **Git**: Tải tại [git-scm.com](https://git-scm.com/).

---

### Các bước thiết lập & chạy dự án từ đầu

#### Bước 1: Clone dự án về máy
```powershell
git clone https://github.com/darian311204/thi-giac-mt.git
cd thi-giac-mt
```

#### Bước 2: Tạo môi trường ảo Python & cài đặt thư viện Backend
```powershell
# Tạo virtualenv
python -m venv venv

# Kích hoạt virtualenv (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Cài đặt toàn bộ thư viện Python từ requirements.txt
pip install -r requirements.txt
```
*(Nếu không dùng `requirements.txt`, có thể chạy trực tiếp: `pip install fastapi "uvicorn[standard]" python-multipart mediapipe opencv-python numpy pillow`)*

#### Bước 3: Khởi chạy Python Backend (Terminal 1)
```powershell
.\venv\Scripts\python.exe main.py
```
- Backend sẽ tự động kiểm tra mô hình `models/hand_landmarker.task` (tự tải nếu chưa có) và chạy tại: `http://127.0.0.1:8008`

#### Bước 4: Cài đặt & khởi chạy React Frontend (Mở Terminal 2)
```powershell
cd thigiacmt
npm install
npm run dev
```
- Mở trình duyệt truy cập: `http://localhost:5173`

#### Bước 5 (Tùy chọn): Kiểm thử nhanh bằng dòng lệnh không cần Camera
```powershell
.\venv\Scripts\python.exe test_no_cam.py
```
