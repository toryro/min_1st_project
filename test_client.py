import argparse, socket, struct, cv2

def send_frame(sock, frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return False
    data = buf.tobytes()
    sock.sendall(struct.pack(">I", len(data)) + data)
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
    ap.add_argument("--source", default=0,
        help="0(기본 웹캠) 또는 동영상 파일 경로")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit("❌ 영상 열기 실패")

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