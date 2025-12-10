# Chess Tracking System (Final Project)

ระบบตรวจจับและบันทึกการเดินหมากรุกจากวิดีโอแบบอัตโนมัติ โดยใช้ Deep Learning (YOLOv8) และ Computer Vision Techniques

## 📋 Features
- **Board Detection**: ค้นหากระดานหมากรุกและทำ Perspective Transform
- **Piece Detection**: ตระหนักรู้หมากแต่ละตัวบนกระดาน
- **Game Tracking**:  บันทึก PGN (Portable Game Notation) และสร้างไฟล์ CSV สำหรับ Submission
- **Auto-Orientation**: หมุนกระดานอัตโนมัติตามตำแหน่งของหมากขาว

## 🛠️ Prerequisites
ต้องติดตั้ง Python 3.8+ และ Libraries ดังต่อไปนี้:

```bash
pip install opencv-python numpy python-chess ultralytics scipy
```

## 📂 File Structure
โครงสร้างไฟล์ที่ระบบต้องการ:

```
.
├── chessboard-model/
│   └── weights/
│       └── best.pt            # Model สำหรับหากระดาน
├── PIECE_MODEL_v8m/
│   └── weights/
│       └── best.pt            # Model สำหรับหาหมาก
├── test_videos/               # โฟลเดอร์เก็บไฟล์วิดีโอทดสอบ
│   ├── 2_Move_rotate_student.mp4
│   └── ...
├── inference.ipynb            # Notebook หลักสำหรับรันโปรเจกต์
├── chess_project.py           # Source Code หลัก (Reference)
└── README.md                  # คู่มือการใช้งาน
```

## 🚀 How to Run (วิธีใช้งาน)

1. **เตรียมไฟล์**: ตรวจสอบว่ามีไฟล์ Model (`.pt`) และ Video อยู่ในโฟลเดอร์ที่กำหนดครบถ้วน
2. **เปิด Notebook**:
   ```bash
   jupyter notebook inference.ipynb
   ```
3. **รันคำสั่ง (Execute)**:
   - กด **Run All** หรือรันทีละ Cell จากบนลงล่าง
4. **ตรวจสอบผลลัพธ์**:
   - ระบบจะสร้างไฟล์ `submission.csv` ที่มีรายละเอียดการเดินหมากของทุกวิดีโอ
   - Format: `row_id` (ชื่อไฟล์วิดีโอ) และ `output` (PGN Moves)

## 💡 System Pipeline

1. **Board Stabilization**: ใช้ YOLO หาขอบกระดานและปรับมุมมอง (Warp)
2. **Orientation Fix**: วิเคราะห์ตำแหน่งหมากเพื่อหมุนกระดานให้ถูกต้อง (White at Bottom/Top)
3. **Grid Projection**: ตีตาราง 8x8 บนภาพที่ Warp แล้ว
4. **State Analysis**: อ่านค่า FEN ในแต่ละเฟรมและแก้ความไม่นิ่ง (Jitter) ด้วย Voting Mechanism
5. **Move validation**: ใช้ `python-chess` ตรวจสอบความถูกต้องของกติกาหมากรุกก่อนบันทึก Move