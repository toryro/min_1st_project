#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thread-Pipelined TCP Client (macOS) – display on **main thread**
"""
from threading import Thread, Event
from queue     import Queue, Empty, Full
import cv2, socket, struct, numpy as np, time, sys

# ────────────── 사용자 설정 ────────────────────────────
SERVER_IP    = "127.0.0.1"
SERVER_PORT  = 9888
#VIDEO_SOURCE = 0                              # 0=웹캠, 또는 "/Users/tory/Videos/test.mp4"
VIDEO_SOURCE = "/Users/tory/Tory/02.Study/movies/test_movie_008.mp4"
JPEG_QUALITY = 80                             # 캡쳐화면 품질(95를 기본으로 하며, 상황에 따라 낮출수도 있다.)
QUEUE_SIZE   = 10
# ──────────────────────────────────────────────────────

def capture_frames(cap, frame_q, stop):
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            stop.set(); break
        try:
            frame_q.put(frame, timeout=.2)
        except Full:
            pass                                # drop if backlog

    cap.release()

def send_and_receive(sock, frame_q, result_q, stop):
    enc_param = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    while not stop.is_set():
        try:
            frame = frame_q.get(timeout=.2)
        except Empty:
            continue

        ok, buf = cv2.imencode(".jpg", frame, enc_param)
        if not ok: continue
        jpeg_bytes = buf.tobytes()

        try:
            sock.sendall(struct.pack(">I", len(jpeg_bytes)))
            sock.sendall(jpeg_bytes)
            resp = sock.recv(4096).decode("utf-8", errors="ignore")
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            stop.set(); break

        bboxes = parse_bbox_string(resp)
        result_q.put((frame, bboxes))

def parse_bbox_string(s: str):
    if not s: return []
    nums = []
    for t in s.strip("[]").split(","):
        t = t.strip()
        if t:
            try: nums.append(int(float(t)))
            except ValueError: pass
    return [tuple(nums[i:i+4]) for i in range(0, len(nums), 4)]

def display_loop(result_q, stop):
    cv2.namedWindow("Client (q to quit)", cv2.WINDOW_NORMAL)
    t0 = time.time(); fcnt = 0; fps = 0.
    while not stop.is_set():
        try:
            frame, bboxes = result_q.get(timeout=.01)
        except Empty:
            # still need to pump waitKey so GUI stays responsive
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop.set(); break
            continue

        for (x, y, w, h) in bboxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        fcnt += 1
        if (now := time.time()) - t0 >= 1.:
            fps, fcnt, t0 = fcnt / (now - t0), 0, now
        cv2.putText(frame, f"FPS {fps:.1f}", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow("Client (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop.set(); break

    cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        sys.exit("❌ 비디오 소스를 열 수 없습니다.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.connect((SERVER_IP, SERVER_PORT))

    frame_q, result_q = Queue(QUEUE_SIZE), Queue(QUEUE_SIZE)
    stop_event = Event()

    # ── 백그라운드 스레드 두 개만 기동 ──
    threads = [
        Thread(target=capture_frames, args=(cap, frame_q, stop_event), daemon=True),
        Thread(target=send_and_receive, args=(sock, frame_q, result_q, stop_event), daemon=True),
    ]
    for t in threads: t.start()

    # ── 디스플레이는 메인 스레드 ──
    display_loop(result_q, stop_event)

    # ── 종료 정리 ──
    stop_event.set()
    for t in threads: t.join()
    sock.close()

if __name__ == "__main__":
    main()