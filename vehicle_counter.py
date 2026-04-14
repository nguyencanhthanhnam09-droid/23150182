import cv2
import math
import time
import os
import requests
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ===== CẤU HÌNH =====
API_URL = "http://127.0.0.1:5000/data"
SAVE_FOLDER = "violations"
DISTANCE_LIMIT = 120  # Ngưỡng khoảng cách (pixel)

# ===== KHỞI TẠO MODEL =====
# 1. Model nhận diện người để theo dõi
person_model = YOLO("yolov8n.pt")  
# 2. Model nhận diện khẩu trang (File bạn vừa tải và đổi tên)
try:
    mask_model = YOLO("mask.pt")
    class_names = mask_model.names 
    print(f"✅ Đã tải model khẩu trang. Danh sách lớp: {class_names}")
except Exception as e:
    print(f"❌ Lỗi: Không tìm thấy file mask.pt trong thư mục. Vui lòng kiểm tra lại! {e}")
    exit()

# Khởi tạo bộ theo dõi đối tượng
tracker = DeepSort(max_age=30)

# ===== KẾT NỐI CAMERA =====
cap = cv2.VideoCapture(0)
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

last_alert_time = 0

print("--- Hệ thống bắt đầu giám sát ---")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không thể đọc dữ liệu từ Camera.")
        break

    frame = cv2.resize(frame, (640, 480))
    
    # BƯỚC 1: Nhận diện người toàn cảnh
    results = person_model(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        if int(box.cls[0]) == 0: # Chỉ lấy class người
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))

    # BƯỚC 2: Cập nhật ID theo dõi người
    tracks = tracker.update_tracks(detections, frame=frame)
    centers, boxes, ids, mask_info = [], [], [], []

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = map(int, track.to_ltrb())
        x1, y1, x2, y2 = max(0, l), max(0, t), min(640, l+w), min(480, t+h)

        # BƯỚC 3: Soi khẩu trang trên vùng ảnh người (Crop)
        face_crop = frame[y1:y2, x1:x2]
        status_label = "NO MASK"
        status_color = (0, 0, 255) # Đỏ mặc định

        if face_crop.size > 0:
            mask_results = mask_model(face_crop, verbose=False)[0]
            for m in mask_results.boxes:
                cls_id = int(m.cls[0])
                c_name = class_names[cls_id].lower()
                
                # Kiểm tra tên class để xác định trạng thái đeo
                # Logic: Nếu tên class có chữ 'mask' và KHÔNG có chữ 'no' -> Có đeo
                if 'mask' in c_name and 'no' not in c_name:
                    status_label = "MASKED"
                    status_color = (0, 255, 0) # Xanh lá
                    break

        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
        boxes.append((x1, y1, x2, y2))
        ids.append(track_id)
        mask_info.append((status_label, status_color))

    # BƯỚC 4: Tính toán khoảng cách giữa mọi người
    violations_dist = set()
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = math.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
            if dist < DISTANCE_LIMIT:
                violations_dist.add(i)
                violations_dist.add(j)

    # BƯỚC 5: Xử lý cảnh báo và gửi dữ liệu về Server Flask
    no_mask_count = sum(1 for m in mask_info if m[0] == "NO MASK")
    
    if len(violations_dist) > 0 or no_mask_count > 0:
        now = time.time()
        if now - last_alert_time > 3: # Giới hạn 3 giây gửi 1 lần
            filename = f"{SAVE_FOLDER}/v_{int(now)}.jpg"
            cv2.imwrite(filename, frame)
            
            payload = {
                "violations_dist": len(violations_dist),
                "no_mask_count": no_mask_count,
                "time": time.strftime("%H:%M:%S"),
                "image_path": filename
            }
            try:
                requests.post(API_URL, json=payload, timeout=1)
                print(f"📢 Đã gửi cảnh báo: {len(violations_dist)} người gần, {no_mask_count} không khẩu trang.")
            except:
                print("⚠️ Server chưa bật hoặc lỗi kết nối.")
            
            last_alert_time = now

    # BƯỚC 6: Vẽ hiển thị lên màn hình Camera
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        txt, m_color = mask_info[i]
        # Màu khung hộp: Đỏ nếu vi phạm khoảng cách, Xanh nếu an toàn
        box_color = (0, 0, 255) if i in violations_dist else (0, 255, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        display_text = f"ID:{ids[i]} | {txt}"
        cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, m_color, 2)

    cv2.imshow("AI COVID MONITORING", frame)

    if cv2.waitKey(1) & 0xFF == 27: # Nhấn ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()