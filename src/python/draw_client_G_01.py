import cv2
import socket
import struct
import numpy as np
import time

# --- 설정 ---
SERVER_IP = '127.0.0.1'  # C++ 서버가 실행 중인 컴퓨터의 IP 주소
SERVER_PORT = 9888
# 비디오 소스 설정: 0 for webcam, 'path/to/video.mp4' for a file
#VIDEO_SOURCE = 0 # 0번 카메라 또는 'test.mp4'와 같은 파일 경로
VIDEO_SOURCE = '/Users/tory/Tory/02.Study/movies/test_movie_001.mp4'

def main():
    # ─── FPS 변수 ───
    fps = 0.0
    frame_cnt = 0
    fps_t0 = time.time()

    # TCP 소켓 생성
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # 서버에 연결 시도
        client_socket.connect((SERVER_IP, SERVER_PORT))
        print(f"INFO: 서버({SERVER_IP}:{SERVER_PORT})에 성공적으로 연결되었습니다.")
    except socket.error as e:
        print(f"ERROR: 서버 연결에 실패했습니다. 서버가 실행 중인지 확인하세요. ({e})")
        return

    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"ERROR: 비디오 소스를 열 수 없습니다: {VIDEO_SOURCE}")
        client_socket.close()
        return

    print("INFO: 클라이언트를 시작합니다. 'q' 키를 누르면 종료됩니다.")

    while cap.isOpened():
        # 프레임 단위로 비디오 읽기
        ret, frame = cap.read()
        if not ret:
            print("INFO: 비디오 스트림의 끝에 도달했거나 오류가 발생했습니다. 프로그램을 종료합니다.")
            break

        # 1. 프레임을 JPEG 형식으로 인코딩 (압축)
        # JPEG 품질을 조절하여 전송량과 화질 사이의 균형을 맞출 수 있습니다. (기본값 95)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
        if not result:
            print("WARNING: 프레임 인코딩에 실패했습니다.")
            continue
        
        # 인코딩된 데이터를 바이트 형태로 변환
        data = encoded_frame.tobytes()

        try:
            # 2. 데이터 크기를 4바이트 unsigned long (빅 엔디안)으로 패킹하여 전송
            # 서버가 수신할 데이터의 길이를 미리 알려주기 위함입니다.
            client_socket.sendall(struct.pack('>L', len(data)))

            # 3. 실제 이미지 데이터 전송
            client_socket.sendall(data)

            # 4. 서버로부터 추론 결과(바운딩 박스 좌표) 수신
            # 서버가 보낼 데이터가 4096 바이트를 넘지 않는다고 가정
            response = client_socket.recv(4096).decode('utf-8')
            
            # 5. 수신한 문자열을 파싱하여 바운딩 박스 그리기
            if response:
                try:
                    # 양쪽 대괄호 '[]' 제거 및 쉼표로 분리
                    coords_str = response.strip('[]').split(',')
                    # 리스트가 비어있지 않고, 첫 번째 요소가 빈 문자열이 아닌 경우에만 처리
                    if coords_str and coords_str[0]:
                        # 문자열 좌표를 정수형으로 변환
                        coords = [int(c.strip()) for c in coords_str]
                        # 4개씩 묶어서 (x, y, w, h) 좌표로 처리
                        for i in range(0, len(coords), 4):
                            x, y, w, h = coords[i:i+4]
                            # 원본 프레임에 녹색 사각형 그리기
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    

                    # ─── FPS 계산 ───
                    frame_cnt += 1
                    elapsed = time.time() - fps_t0
                    if elapsed >= 1.0:
                        fps = frame_cnt / elapsed
                        fps_t0, frame_cnt = time.time(), 0
                    cv2.putText(frame, f"FPS {fps:.1f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)

                except (ValueError, IndexError) as e:
                    print(f"WARNING: 서버로부터 받은 좌표 파싱 중 오류 발생: {e}, 수신 데이터: '{response}'")

        except socket.error as e:
            print(f"ERROR: 소켓 통신 중 오류가 발생했습니다: {e}")
            break # 통신 오류 발생 시 루프 종료


        # 결과 프레임을 화면에 표시
        cv2.imshow('Client View', frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # 자원 해제
    print("INFO: 자원을 해제하고 클라이언트를 종료합니다.")
    cap.release()
    client_socket.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()