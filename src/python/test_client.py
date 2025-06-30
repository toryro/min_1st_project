import argparse, socket, struct, cv2

def send_frame(sock, frame):
    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        print("⚠️ 프레임 인코딩 실패")
        return False

    data = buffer.tobytes()
    try:
        sock.sendall(struct.pack(">I", len(data)) + data)
    except BrokenPipeError:
        print("❌ 서버가 연결을 끊었습니다 (Broken pipe)")
        return False
    return True

def recv_bbox(sock):
    raw = b''
    while len(raw) < 16:                 # 4 * int32
        chunk = sock.recv(16 - len(raw))
        if not chunk:
            return None
        raw += chunk
    return struct.unpack(">iiii", raw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=0, help="0(기본 웹캠) 또는 동영상 파일 경로")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9888)
    args = ap.parse_args()

    # ① 기본값은 문자열 "0"
    src = args.source
    # ② 숫자형 문자열이면 int 로 변환 → 웹캠 index
    if isinstance(src, str) and src.isdigit():
        src = int(src)          # ex) "0" → 0
    # ③ 아니면 파일 경로 그대로
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"⚠️  영상을 열 수 없습니다: {args.source}")
        exit(1)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((args.host, args.port))
        print("✅ 서버 연결 완료")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not send_frame(s, frame):
                break

            bbox = recv_bbox(s)
            if bbox is None:
                print("서버 연결 끊김"); break
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.imshow("Client – with bbox", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()