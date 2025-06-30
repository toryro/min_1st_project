import argparse, socket, struct, cv2, time

def send_frame(sock, frame, quality=80):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok: return False
    data = buf.tobytes()
    sock.sendall(struct.pack('>I', len(data)) + data)
    return True

def recv_until_newline(sock):
    data = b''
    while True:
        ch = sock.recv(1)
        if not ch: return None
        if ch == b'\n': break
        data += ch
    return data.decode()

def parse_boxes(s):
    dets = []
    if not s: return dets
    for item in s.split(';'):
        if not item: continue
        x1,y1,x2,y2,c,conf = item.split(',')
        dets.append((int(x1),int(y1),int(x2),int(y2),int(c),float(conf)))
    return dets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=9888)
    ap.add_argument('--source', default='0',
                    help='카메라 인덱스(숫자) 또는 동영상 파일 경로')
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print('⚠️  영상 소스 열기 실패'); return

    sock = socket.create_connection((args.host, args.port))
    print('✅ 서버 연결 완료')

    while True:
        ret, frame = cap.read()
        if not ret: break
        if not send_frame(sock, frame):
            print('⚠️  전송 실패'); break

        msg = recv_until_newline(sock)
        if msg is None: print('서버 연결 종료'); break
        boxes = parse_boxes(msg)

        for x1,y1,x2,y2,cls,conf in boxes:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            label = f'{cls}:{conf:.2f}'
            cv2.putText(frame, label, (x1, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow('Client Display', frame)
        if cv2.waitKey(1) & 0xFF == 27: break     # ESC to quit
    cap.release(); sock.close()

if __name__ == '__main__':
    main()