#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thread-Pipelined TCP Client (macOS) – display on **main thread**
"""
from threading import Thread, Event
from queue     import Queue, Empty, Full
import cv2, socket, struct, numpy as np, time, sys
import re
import json, pathlib

# ────────────── 사용자 설정 ────────────────────────────
JPEG_QUALITY = 80                             # 캡쳐화면 품질(95를 기본으로 하며, 상황에 따라 낮출수도 있다.)
QUEUE_SIZE   = 10

# ─── 설정 읽기 ───
cfg_path = pathlib.Path("/Users/tory/Tory/02.Study/01.1team/min_1st_project/draw_config.json")
cfg = json.loads(cfg_path.read_text())

SERVER_IP    = cfg["client"]["server_ip"]
SERVER_PORT  = cfg["client"]["server_port"]
VIDEO_SOURCE = cfg["client"]["video_source"]
# ──────────────────────────────────────────────────────

CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench",
    "toothbrush"
    #"big vehicle","vehicle","bike","human","animal","obstacle", # 우리 프로젝트에서 추가한것.
]

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
    rx_buf = ""                                   # ← 수신 버퍼

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
            try:
                resp = sock.recv(4096).decode("utf-8", errors="ignore")
                if not resp:
                    stop.set(); break
                rx_buf += resp
            except OSError:
                stop.set(); break
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            stop.set(); break

        # \n 기준으로 완전한 한 줄씩 처리
        while "\n" in rx_buf:
            line, rx_buf = rx_buf.split("\n", 1)
            bboxes = parse_bbox_string(line)
            result_q.put((frame, bboxes))

def parse_bbox_string(s: str):
    if not s: return []

    # ① 숫자(정수·실수·음수)만 추출
    nums = [int(float(n))          # '12.3' 도 안전 변환
            for n in re.findall(r"-?\d+\.?\d*", s)]

    # ② 5-튜플(클래스 포함) or 4-튜플 모두 수용
    step = 5 if len(nums) % 5 == 0 else 4
    return [tuple(nums[i:i+step]) for i in range(0, len(nums), step)]

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

        for bbox in bboxes:
            if len(bbox) == 5:
                x, y, w, h, cls_id = bbox
            else:                          # 호환용(클래스 없음)
                x, y, w, h = bbox
                cls_id = None

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if cls_id is not None:
                label = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
                cv2.putText(frame, label, (x, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                            cv2.LINE_AA)

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