import cv2
from ultralytics import YOLO
import time  # FPS 측정을 위한 모듈 추가

"""
YOLOv8 Distance Estimation + Correction + FPS 표시
─────────────────────────────────────────────────
원본 코드에 FPS 계산·표시 기능을 추가했습니다.
• 한 프레임 처리 시간 측정 → 1초마다 FPS 업데이트
• 좌상단(30,30)에 "FPS 72.4" 형식으로 오버레이
"""

# 모델 로딩 (경로는 필요에 따라 수정)
model = YOLO('./models/weights14/best.pt')
#model = YOLO('/Users/tory/Tory/02.Study/01.1team/models/yolov8n.pt')

# 테스트 영상 경로
#video_path = '/Users/tory/Tory/02.Study/movies/output_part.webm'
video_path = '/Users/tory/Tory/02.Study/movies/test_movie_007.mp4'
cap = cv2.VideoCapture(video_path)

# 클래스별 실제 크기 (height, width) in meters
object_sizes = {
    'person': (1.7, 0.5),
    'car': (1.5, 1.8),
    'truck': (3.5, 2.5),
    'bus': (3.2, 2.5),
    'bicycle': (1.2, 0.5),
    'motorcycle': (1.4, 0.6),
    'dog': (0.5, 0.3),
}

reference_width = 640     # ref px
reference_focal = 700     # ref focal

def distance_to_y(distance_m, frame_height):
    return int(frame_height - distance_m * (frame_height / 5.0))

# 첫 프레임 → 해상도 체크
ret, frame = cap.read()
if not ret:
    raise RuntimeError('❌ Cannot read first frame')
H, W = frame.shape[:2]
focal_length = int(reference_focal * (W / reference_width))
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

a, b = 0.75, -0.5   # 보정 파라미터

# ─── FPS 변수 ───
fps = 0.0
frame_cnt = 0
fps_t0 = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        pix_h = max(y2 - y1, 1)
        pix_w = max(x2 - x1, 1)
        real_h, real_w = object_sizes.get(label, (1.7, 0.5))

        d_h = (real_h * focal_length) / pix_h
        d_w = (real_w * focal_length) / pix_w
        dist = a * ((d_h + d_w) / 2) + b

        color = (0,0,255) if dist < 1.5 else (0,255,255) if dist < 2.0 else (0,255,0)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"{dist:.2f}m", (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 위험 영역 선(단순화)
    overlay = frame.copy()
    y_1m = distance_to_y(0.8, H)
    y_2m = distance_to_y(1.1, H)
    cv2.rectangle(overlay, (0, y_1m), (W, H), (0,0,255), -1)
    cv2.rectangle(overlay, (0, y_2m), (W, y_1m), (0,255,255), -1)
    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

    # ─── FPS 계산 ───
    frame_cnt += 1
    elapsed = time.time() - fps_t0
    if elapsed >= 1.0:
        fps = frame_cnt / elapsed
        fps_t0, frame_cnt = time.time(), 0

    cv2.putText(frame, f"FPS {fps:.1f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)

    cv2.imshow('YOLOv8 Distance + FPS', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()