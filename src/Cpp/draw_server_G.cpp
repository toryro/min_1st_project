#include <iostream>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h> // 실제 ONNX Runtime 사용 시 주석 해제

#define PORT 9888

// --- 중요 ---
// 이 함수는 실제 ONNX 모델로 추론을 수행하는 로직으로 대체해야 합니다.
// 현재는 더미 좌표를 반환합니다.
std::string run_inference_on_image(const cv::Mat& image) {
    // -----------------------------------------------------------------
    // 여기에 ONNX Runtime C++ API를 사용한 추론 코드를 작성합니다.
    //
    // 1. Ort::Env, Ort::SessionOptions, Ort::Session 생성 및 모델 로드
    // 2. 입력 이미지 전처리 (리사이즈, 정규화, HWC to CHW 변환 등)
    // 3. Ort::Value 입력 텐서 생성
    // 4. session.Run() 으로 추론 실행
    // 5. 출력 텐서에서 결과(바운딩 박스 좌표, 클래스, 신뢰도 등) 추출
    // 6. 후처리 (NMS 등)
    // 7. 결과를 "[x1, y1, w1, h1, x2, y2, w2, h2, ...]" 형태의 문자열로 변환
    // -----------------------------------------------------------------

    std::cout << "가상 추론 실행. 이미지 크기: " << image.size() << std::endl;

    // 예시 더미 결과: (10,20)에서 50x60 크기의 박스, (100,120)에서 80x90 크기의 박스
    return "[10, 20, 50, 60, 100, 120, 80, 90]";
}

int main() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    std::cout << "서버가 " << PORT << " 포트에서 대기 중..." << std::endl;

    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }

    std::cout << "클라이언트 연결됨." << std::endl;

    while (true) {
        // 1. 이미지 데이터 크기 수신 (4바이트)
        uint32_t data_size;
        int bytes_read = read(new_socket, &data_size, sizeof(data_size));
        if (bytes_read <= 0) {
            std::cout << "클라이언트 연결 끊김." << std::endl;
            break;
        }
        data_size = ntohl(data_size); // Network to Host byte order

        // 2. 이미지 데이터 수신
        std::vector<uchar> img_buffer(data_size);
        bytes_read = 0;
        while (bytes_read < data_size) {
            int result = read(new_socket, img_buffer.data() + bytes_read, data_size - bytes_read);
            if (result <= 0) {
                 std::cout << "데이터 수신 중 연결 끊김." << std::endl;
                 close(new_socket);
                 return 1;
            }
            bytes_read += result;
        }
        
        // 3. 수신한 버퍼를 이미지로 디코딩
        cv::Mat image = cv::imdecode(img_buffer, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "이미지 디코딩 실패" << std::endl;
            continue;
        }

        // 4. 추론 실행
        std::string bounding_boxes_str = run_inference_on_image(image);

        // 5. 결과 전송
        send(new_socket, bounding_boxes_str.c_str(), bounding_boxes_str.length(), 0);
        std::cout << "결과 전송: " << bounding_boxes_str << std::endl;
    }

    close(new_socket);
    close(server_fd);

    return 0;
}