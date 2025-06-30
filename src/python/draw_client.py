import socket
import cv2
import struct
import numpy as np

SERVER_IP = '127.0.0.1'
SERVER_PORT = 9888

# 로컬 영상 열기
video_path = '/Users/tory/Tory/02.Study/movies/test_movie_003.mp4'  # 🔁 원하는 영상 파일명으로 변경
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ 비디오 파일을 열 수 없습니다.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ 영상 끝")
        break

    # JPEG 인코딩
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    # 소켓 연결 및 이미지 전송
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((SERVER_IP, SERVER_PORT))
        sock.sendall(img_bytes)

        # 바운딩박스 수신
        bbox_data = b""
        while len(bbox_data) < 16:
            packet = sock.recv(16 - len(bbox_data))
            if not packet:
                break
            bbox_data += packet
        sock.close()
    except Exception as e:
        print("❌ 서버 통신 오류:", e)
        continue

    if len(bbox_data) == 16:
        x1, y1, x2, y2 = struct.unpack('4f', bbox_data)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, "Detected", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    else:
        print("⚠️ 바운딩박스 수신 실패")

    # 결과 프레임 표시
    cv2.imshow("Detection Result", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()