import cv2
import socket
import struct
import numpy as np

# --- 설정 ---
SERVER_IP = '127.0.0.1'  # 서버 IP 주소 (Mac의 IP 주소)
SERVER_PORT = 9888
# 비디오 소스 설정: 0 for webcam, or 'path/to/video.mp4' for a file
#VIDEO_SOURCE = 0 # 0번 카메라 사용. 'test.mp4' 와 같이 파일 경로 지정 가능
VIDEO_SOURCE = '/Users/tory/Tory/02.Study/movies/test_movie_003.mp4'

def main():
    # TCP 소켓 생성 및 서버 연결
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((SERVER_IP, SERVER_PORT))
        print(f"{SERVER_IP}:{SERVER_PORT} 서버에 연결되었습니다.")
    except socket.error as e:
        print(f"서버 연결에 실패했습니다: {e}")
        return

    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"비디오 소스를 열 수 없습니다: {VIDEO_SOURCE}")
        client_socket.close()
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다. 스트림의 끝이거나 오류입니다.")
            break

        # 1. 프레임을 JPEG로 인코딩
        # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90] # JPEG 품질 설정 (기본값 95)
        result, encoded_frame = cv2.imencode('.jpg', frame)
        if not result:
            print("프레임 인코딩에 실패했습니다.")
            continue
        
        data = encoded_frame.tobytes()

        try:
            # 2. 데이터 크기를 4바이트 unsigned long으로 패킹하여 전송 (Network Byte Order)
            client_socket.sendall(struct.pack('>L', len(data)))
            # 3. 실제 이미지 데이터 전송
            client_socket.sendall(data)

            # 4. 서버로부터 추론 결과 수신
            response = client_socket.recv(1024).decode('utf-8')
            
            # 5. 수신한 문자열 파싱하여 바운딩 박스 그리기
            # 서버가 보낸 형식: "[x1, y1, w1, h1, x2, y2, w2, h2, ...]"
            try:
                # 괄호 제거 및 쉼표로 분리
                coords_str = response.strip('[]').split(',')
                if coords_str and coords_str[0]: # 빈 문자열이 아닐 경우
                    # 4개씩 묶어서 좌표 추출
                    coords = [int(c.strip()) for c in coords_str]
                    for i in range(0, len(coords), 4):
                        x, y, w, h = coords[i:i+4]
                        print(f"rectangle : {x}, {y}, {w}, {h}")
                        # 프레임에 사각형 그리기
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            except (ValueError, IndexError) as e:
                print(f"좌표 파싱 오류: {e}, 수신 데이터: {response}")

        except socket.error as e:
            print(f"소켓 통신 오류: {e}")
            break

        # 결과 화면에 표시
        cv2.imshow('Client View', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # 자원 해제
    cap.release()
    client_socket.close()
    cv2.destroyAllWindows()
    print("클라이언트 종료.")

if __name__ == '__main__':
    main()