import cv2
import numpy as np
import csv
import chess
import chess.pgn
from ultralytics import YOLO
from collections import deque, Counter 
from scipy.signal import find_peaks 

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
BOARD_MODEL_PATH = 'chessboard-model/weights/best.pt'
PIECE_MODEL_PATH = 'PIECE_MODEL_v8m/weights/best.pt'
# VIDEO_PATH = 'test_videos/2_Move_rotate_student.mp4'
# VIDEO_PATH = 'test_videos/4_Move_studet.mp4'
VIDEO_PATH = 'test_videos/6_Move_student.mp4'

# 🔥 NEW: เริ่มต้นที่วินาทีที่เท่าไหร่
START_SEC = 0

# 🔥 Logic เดิมของคุณ (ห้ามแก้)
STABILITY_THRESHOLD = 30  

CLASS_TO_FEN = {
    0: 'b', 1: 'k', 2: 'n', 3: 'p', 4: 'q', 5: 'r',
    6: 'B', 7: 'K', 8: 'N', 9: 'P', 10: 'Q', 11: 'R'
}

# ==========================================
# 🛠️ HELPER CLASSES (Original Logic)
# ==========================================

class BoardStabilizer:
    def __init__(self, alpha=0.2, max_dist=50):
        self.prev_points = None; self.alpha = alpha; self.max_dist = max_dist
    def update(self, current_points):
        if self.prev_points is None: self.prev_points = current_points; return current_points
        dist = np.linalg.norm(current_points - self.prev_points)
        if dist > self.max_dist: return self.prev_points
        smoothed = (current_points * self.alpha) + (self.prev_points * (1 - self.alpha))
        self.prev_points = smoothed
        return smoothed

class GridProjectionCalibrator:
    def __init__(self):
        self.accumulated_edges_v = None; self.accumulated_edges_h = None
        self.frame_count = 0; self.is_calibrated = False
        self.grid_params = (20, 20, 75, 75) 
    def add_frame(self, warped_image):
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        sob_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sob_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        if self.accumulated_edges_v is None:
            self.accumulated_edges_v = cv2.convertScaleAbs(sob_x).astype("float32")
            self.accumulated_edges_h = cv2.convertScaleAbs(sob_y).astype("float32")
        else:
            cv2.accumulateWeighted(cv2.convertScaleAbs(sob_x).astype("float32"), self.accumulated_edges_v, 0.1)
            cv2.accumulateWeighted(cv2.convertScaleAbs(sob_y).astype("float32"), self.accumulated_edges_h, 0.1)
        self.frame_count += 1
    def compute_grid(self):
        if self.frame_count < 10: return self.fallback_grid(640, 640)
        proj_x = np.sum(self.accumulated_edges_v, axis=0)
        proj_y = np.sum(self.accumulated_edges_h, axis=1)
        peaks_x, _ = find_peaks(cv2.normalize(proj_x, None, 0, 255, cv2.NORM_MINMAX).flatten(), height=40, distance=40)
        peaks_y, _ = find_peaks(cv2.normalize(proj_y, None, 0, 255, cv2.NORM_MINMAX).flatten(), height=40, distance=40)
        if len(peaks_x) < 7 or len(peaks_y) < 7: return self.fallback_grid(640, 640)
        start_x, end_x = peaks_x[0], peaks_x[-1]
        start_y, end_y = peaks_y[0], peaks_y[-1]
        self.grid_params = (start_x, start_y, (end_x - start_x)/8.0, (end_y - start_y)/8.0)
        self.is_calibrated = True; return True
    def fallback_grid(self, w, h):
        margin = w * 0.04; cell = (w - 2*margin) / 8.0
        self.grid_params = (margin, margin, cell, cell)
        self.is_calibrated = True; return True

class ChessGameTracker:
    def __init__(self):
        self.board = None; self.pgn_moves = []
        self.grid_buffer = deque(maxlen=15)
        self.last_stable_fen = None; self.candidate_fen = None; self.stability_counter = 0
        self.black_started = False 

    def board_to_fen_part(self, board_grid):
        fen_rows = []
        for row in range(8):
            empty = 0; fen = ""
            for col in range(8):
                piece = board_grid[row][col]
                if piece == '': empty += 1
                else:
                    if empty > 0: fen += str(empty); empty = 0
                    fen += piece
            if empty > 0: fen += str(empty)
            fen_rows.append(fen)
        return "/".join(fen_rows)

    def update(self, current_grid):
        simple_grid = [['' for _ in range(8)] for _ in range(8)]
        for r in range(8):
            for c in range(8):
                if current_grid[r][c]: simple_grid[r][c] = current_grid[r][c]['fen']
        self.grid_buffer.append(simple_grid)
        if len(self.grid_buffer) < 5: return None
        stable_grid = [['' for _ in range(8)] for _ in range(8)]
        for r in range(8):
            for c in range(8):
                candidates = [grid[r][c] for grid in self.grid_buffer]
                from collections import Counter
                most_common, count = Counter(candidates).most_common(1)[0]
                if count >= len(self.grid_buffer) * 0.6: stable_grid[r][c] = most_common
                else: return None 

        detected_fen = self.board_to_fen_part(stable_grid)
        if self.board is None:
            print(f"[INIT] Start Position: {detected_fen}")
            self.board = chess.Board(detected_fen + " w KQkq - 0 1")
            self.last_stable_fen = detected_fen
            return None

        if detected_fen == self.last_stable_fen:
            self.stability_counter = 0; self.candidate_fen = None; return None
        if detected_fen == self.candidate_fen: self.stability_counter += 1
        else: self.candidate_fen = detected_fen; self.stability_counter = 0
            
        if self.stability_counter >= STABILITY_THRESHOLD:
            move_san = self.validate_and_push_move(self.candidate_fen)
            if move_san: self.last_stable_fen = self.candidate_fen
            self.stability_counter = 0; self.candidate_fen = None
            return move_san
        return None

    def validate_and_push_move(self, target_fen_part):
        target_dict = self.fen_to_dict(target_fen_part)
        candidates = []
        turns_to_check = [self.board.turn, not self.board.turn]
        for turn in turns_to_check:
            original_turn = self.board.turn
            self.board.turn = turn
            for move in self.board.legal_moves:
                move_san = self.board.san(move)
                from_sq, to_sq = move.from_square, move.to_square
                # Map using "White Top" Logic (Standard for your setup)
                src_r, src_c = 7 - (from_sq // 8), from_sq % 8
                dst_r, dest_c = 7 - (to_sq // 8), to_sq % 8
                
                moving_piece = self.board.piece_at(from_sq).symbol()
                detected_at_src = target_dict.get((src_r, src_c))
                detected_at_dest = target_dict.get((dst_r, dest_c))
                
                score = 0
                if detected_at_dest == moving_piece: score += 20
                elif detected_at_dest is not None: score += 5 
                else: score -= 50 
                if detected_at_src is None: score += 20
                else: score -= 20 
                if score > 0: candidates.append({'move': move, 'san': move_san, 'score': score, 'turn': turn})
            self.board.turn = original_turn 
            
        if not candidates: return None
        best = sorted(candidates, key=lambda x: x['score'], reverse=True)[0]
        if best['score'] >= 30:
            if len(self.pgn_moves) == 0 and best['turn'] == chess.BLACK:
                print("⚫ DETECTED: Black moves first!")
                self.black_started = True
            if best['turn'] != self.board.turn: self.board.turn = best['turn']
            self.board.push(best['move'])
            self.pgn_moves.append(best['san'])
            print(f"🚀 CONFIRMED MOVE: {best['san']} (Score: {best['score']})")
            return best['san']
        return None

    def fen_to_dict(self, fen_str):
        m = {}; rows = fen_str.split('/')
        for r, row in enumerate(rows):
            c = 0
            for char in row:
                if char.isdigit(): c += int(char)
                else: m[(r,c)] = char; c += 1
        return m
    
    def save_pgn(self, filename="game_result.pgn"):
        print(f"\n💾 Saving PGN to {filename}...")
        pgn_str = ""; idx, move_num = 0, 1
        if self.black_started:
            if idx < len(self.pgn_moves):
                pgn_str += f"1... {self.pgn_moves[idx]} "; idx += 1; move_num = 2 
        while idx < len(self.pgn_moves):
            pgn_str += f"{move_num}. {self.pgn_moves[idx]} "; idx += 1
            if idx < len(self.pgn_moves): pgn_str += f"{self.pgn_moves[idx]} "; idx += 1
            move_num += 1
        with open(filename, "w") as f: f.write(pgn_str.strip())
        print("✅ PGN Saved.")

    def save_csv(self, filename="submission_test.csv"):
        print(f"\n📊 Saving CSV to {filename}...")
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Move_ID", "Move_SAN"])
            for i, move in enumerate(self.pgn_moves): writer.writerow([i + 1, move])
        print("✅ CSV Saved.")

# ==========================================
# 📐 GEOMETRY & DETECTION
# ==========================================

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def warp_chessboard(image, board_model, fixed_M=None, stabilizer=None):
    if fixed_M is not None: return cv2.warpPerspective(image, fixed_M, (640, 640)), fixed_M
    results = board_model(image, verbose=False)
    if not results[0].masks: return None, None
    mask = cv2.resize(results[0].masks.data[0].cpu().numpy().astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None
    largest = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest, 0.04 * cv2.arcLength(largest, True), True)
    if len(approx) != 4: approx = cv2.boxPoints(cv2.minAreaRect(largest))
    src_pts = order_points(approx.reshape(4, 2).astype("float32"))
    if stabilizer: src_pts = stabilizer.update(src_pts)
    M = cv2.getPerspectiveTransform(src_pts, np.array([[0,0],[639,0],[639,639],[0,639]], dtype="float32"))
    return cv2.warpPerspective(image, M, (640, 640)), M

# 🔥 HERO FUNCTION: Distance-Based Orientation (แก้ปัญหา 90/45 องศา)
def fix_board_orientation(image, piece_model):
    results = piece_model(image, conf=0.3, verbose=False)
    wx, wy = [], []
    h, w = image.shape[:2]
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        if cls_id in CLASS_TO_FEN and CLASS_TO_FEN[cls_id].isupper():
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            wx.append((x1 + x2)/2)
            wy.append((y1 + y2)/2)
            
    if len(wx) < 3: return image, "Unknown"

    avg_x = np.mean(wx) / w
    avg_y = np.mean(wy) / h
    
    # 🎯 Force "White on TOP" (เพื่อให้เข้ากับ Logic เก่าของคุณ)
    # เช็คว่าหมากขาวอยู่ใกล้ขอบไหนที่สุด แล้วหมุนให้ไปอยู่ 'บน'
    dist_top = (avg_x - 0.5)**2 + (avg_y - 0.15)**2
    dist_bottom = (avg_x - 0.5)**2 + (avg_y - 0.85)**2
    dist_left = (avg_x - 0.15)**2 + (avg_y - 0.5)**2
    dist_right = (avg_x - 0.85)**2 + (avg_y - 0.5)**2
    
    min_dist = min(dist_top, dist_bottom, dist_left, dist_right)
    
    if min_dist == dist_top:
        return image, "Top (Correct)"
    elif min_dist == dist_bottom:
        return cv2.rotate(image, cv2.ROTATE_180), "Bottom -> Rotated 180"
    elif min_dist == dist_left:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), "Left -> Rotated CW"
    elif min_dist == dist_right:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), "Right -> Rotated CCW"
        
    return image, "Ambiguous"

def generate_grid_auto(warped_image, piece_model, grid_params):
    results = piece_model(warped_image, conf=0.5, verbose=False)
    debug_visual = warped_image.copy()
    start_x, start_y, cell_w, cell_h = grid_params
    
    for i in range(9):
        p = int(start_x + (i * cell_w))
        cv2.line(debug_visual, (p, 0), (p, 640), (0, 255, 255), 1)
        p = int(start_y + (i * cell_h))
        cv2.line(debug_visual, (0, p), (640, p), (0, 255, 255), 1)

    board_grid = [[None for _ in range(8)] for _ in range(8)]

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0].item())
        if cls_id not in CLASS_TO_FEN: continue
        
        anchor_x = (x1 + x2) / 2
        anchor_y = y2 - ((y2 - y1) * 0.10) 
        
        col_idx = int((anchor_x - start_x) // cell_w)
        row_idx = int((anchor_y - start_y) // cell_h)
        
        if 0 <= col_idx < 8 and 0 <= row_idx < 8:
            # 🔥 Logic เดิมของคุณ: 
            # ถ้า White อยู่ Top (จากการหมุน) -> เราต้องกลับหัวพิกัดให้เป็น Rank 1
            final_r, final_c = 7 - row_idx, 7 - col_idx 
            
            curr = board_grid[final_r][final_c]
            if curr is None or float(box.conf[0]) > curr['conf']:
                board_grid[final_r][final_c] = {'fen': CLASS_TO_FEN[cls_id], 'conf': float(box.conf[0])}

            cv2.circle(debug_visual, (int(anchor_x), int(anchor_y)), 5, (0,0,255), -1)
            cv2.putText(debug_visual, CLASS_TO_FEN[cls_id], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    return board_grid, debug_visual

# ==========================================
# 🚀 MAIN
# ==========================================
print("⏳ Loading Models...")
board_model = YOLO(BOARD_MODEL_PATH)
piece_model = YOLO(PIECE_MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# 🔥 SKIP TIME LOGIC
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame_idx = int(START_SEC * fps)
if start_frame_idx > 0:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    print(f"⏩ Skipped to {START_SEC}s (Frame {start_frame_idx})")

tracker = ChessGameTracker()
calibrator = GridProjectionCalibrator() 
stabilizer = BoardStabilizer()

fixed_M = None
calibrated = False
locked_rotation = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    warped_img, M = warp_chessboard(frame, board_model, fixed_M=fixed_M, stabilizer=stabilizer)
    
    if warped_img is not None:
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if not calibrated:
            # 1. หมุนภาพให้เป็น White Top (เพื่อให้เข้ากับ Logic เดิม)
            rotated_img, status = fix_board_orientation(warped_img, piece_model)
            calibrator.add_frame(rotated_img)
            
            cv2.putText(frame, f"Calibrating... {frame_idx}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.putText(frame, f"Angle: {status}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Debug (Rotated)", rotated_img)
            
            if frame_idx >= (start_frame_idx + 45):
                success = calibrator.compute_grid() 
                fixed_M = M
                calibrated = True
                
                # จำค่าการหมุนไว้ใช้ยาวๆ
                if "180" in status: locked_rotation = cv2.ROTATE_180
                elif "CCW" in status: locked_rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
                elif "CW" in status: locked_rotation = cv2.ROTATE_90_CLOCKWISE
                else: locked_rotation = None
                
                print(f"🔒 System Locked. Mode: {status}")
        else:
            final_img = warped_img
            if locked_rotation is not None:
                final_img = cv2.rotate(warped_img, locked_rotation)
                
            # ส่งภาพ White Top เข้า Logic เดิม (ซึ่งมันชอบ White Top)
            board_grid, debug_img = generate_grid_auto(final_img, piece_model, calibrator.grid_params)
            move = tracker.update(board_grid)
            
            cv2.putText(frame, f"Moves: {len(tracker.pgn_moves)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            if move: cv2.putText(frame, f"Last: {move}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
            elif tracker.candidate_fen:
                bar = int((tracker.stability_counter / STABILITY_THRESHOLD) * 200)
                cv2.rectangle(frame, (30, 120), (30+bar, 140), (0,255,255), -1)

            cv2.imshow("Debug (Rotated)", debug_img)
            cv2.imshow("Main", frame)

    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
tracker.save_pgn("game_result.pgn")
tracker.save_csv("submission_test.csv")
print("\n✅ DONE.")