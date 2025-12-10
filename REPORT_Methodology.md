# รายงานส่วนที่ 1: ระเบียบวิธีวิจัย (Methodology Report)

**Project**: Chess Tracking System using Computer Vision
**Submitted by**: [ใส่ชื่อของคุณที่นี่]

---

## 1. ภาพรวมของระบบ (System Overview)
ระบบนี้ถูกออกแบบมาเพื่อแปลงวิดีโอการเดินหมากรุกให้เป็นบันทึกเกม (PGN) โดยอัตโนมัติ โดยใช้กระบวนการทำงานแบบ Pipeline ที่ประกอบด้วย Deep Learning Models และ Image Processing Logic

## 2. การตรวจจับกระดาน (Board Detection & Warping)
ใช้โมเดล **YOLOv8** ที่เทรนมาเพื่อตรวจจับขอบกระดานหมากรุกโดยเฉพาะ
- **Input**: เฟรมภาพจากวิดีโอ
- **Process**:
    1. โมเดลทำนาย Mask ของกระดาน
    2. ใช้ `cv2.findContours` หาขอบเขตสี่เหลี่ยม
    3. เรียงจุดมุมทั้ง 4 (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    4. ใช้ `cv2.getPerspectiveTransform` และ `cv2.warpPerspective` เพื่อดึงกระดานให้เป็นภาพ 2D (Top-down view) ขนาด 640x640 pixel

**Technique**: มีการใช้ **Stabilizer (Exponential Moving Average)** เพื่อลดอาการสั่นของมุมกระดานระหว่างเฟรม ทำให้ภาพนิ่งขึ้น

## 3. การตรวจจับตัวหมาก (Piece Detection)
ใช้โมเดล **YOLOv8m** (Medium) ที่เทรนกับ Dataset หมากรุก เพื่อระบุประเภท (Class) และตำแหน่งของหมาก
- **Classes**: 12 Classes (Pawn, Rook, Knight, Bishop, Queen, King ทั้งขาวและดำ)
- **Mapping**: แปลง Class ID เป็น FEN Symbol (เช่น `0 -> b` (Black Bishop), `7 -> K` (White King))

**Technique**: ใช้ **Distance-Based Orientation** ในการเช็คทิศทางกระดาน โดยดูว่ากลุ่มของหมากขาว (ตัวพิมพ์ใหญ่) อยู่ใกล้ขอบด้านไหนของภาพมากที่สุด แล้วหมุนภาพให้อยู่ในทิศที่ถูกต้อง (White at Bottom/Top)

## 4. การวิเคราะห์สถานะเกม (Game State Analysis)
### 4.1 Grid Projection
ระบบไม่ได้ใช้เส้นตารางคงที่ แต่ใช้ **Dynamic Grid Calibration**:
- หา Sobel Edges ของภาพกระดาน
- ใช้ Projection Histogram (รวมค่า Pixel แนวตั้ง/แนวนอน) เพื่อหาตำแหน่งเส้นตาราว
- สร้างตาราง 8x8 ที่แม่นยำแม้กระดานจะเอียงเล็กน้อย

### 4.2 Voting Mechanism
เพื่อป้องกัน False Positive ระบบจะเก็บ History ของ Grid ล่าสุด 15 เฟรม
- ใช้ **Majority Voting** เพื่อหาค่าที่นิ่งที่สุดในแต่ละช่อง (ต้องเหมือนกันอย่างน้อย 60% ของ Buffer)
- หากค่าเปลี่ยนไปและนิ่งติดต่อกัน 30 เฟรม (`STABILITY_THRESHOLD`) ระบบจึงจะยอมรับว่ามีการเดินหมากจริง

### 4.3 Move Validation
ใช้ Library `python-chess` ในการตรวจสอบความถูกต้อง:
- เปรียบเทียบ FEN ปัจจุบันกับ FEN ก่อนหน้า
- หาความแตกต่าง (Diff) ว่าหมากตัวไหนหายไปและตัวไหนย้ายมาใหม่
- ตรวจสอบกับ `board.legal_moves` เพื่อยืนยันว่าการเดินนั้นถูกต้องตามกติกาจริง (ป้องกันการบันทึกมั่ว)
