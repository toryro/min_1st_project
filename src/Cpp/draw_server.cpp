/*
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#define PORT 9888
#define BUFFER_SIZE 65536

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

std::vector<float> run_onnx_inference(const cv::Mat& input_img) {
    // 여기서는 단순히 더미 데이터를 리턴하는 걸로 처리함
    // 실제 ONNX 모델 추론은 이 부분에 구현
    return {0.1f, 0.2f, 0.4f, 0.5f};  // xmin, ymin, xmax, ymax
}

void handle_client(int client_sock) {
    char buffer[BUFFER_SIZE];
    int bytes_received = recv(client_sock, buffer, BUFFER_SIZE, 0);
    if (bytes_received <= 0) {
        std::cerr << "클라이언트로부터 데이터를 받지 못함\n";
        return;
    }

    std::vector<uchar> img_data(buffer, buffer + bytes_received);
    cv::Mat img = cv::imdecode(img_data, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "이미지 디코딩 실패\n";
        return;
    }

    std::vector<float> bbox = run_onnx_inference(img);

    // 바운딩박스 데이터 전송
    send(client_sock, bbox.data(), bbox.size() * sizeof(float), 0);
}

int main() {
    int server_fd, client_sock;
    struct sockaddr_in address;
    socklen_t addrlen = sizeof(address);

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    bind(server_fd, (struct sockaddr*)&address, sizeof(address));
    listen(server_fd, 3);
    std::cout << "서버 대기 중...\n";

    while (true) {
        client_sock = accept(server_fd, (struct sockaddr*)&address, &addrlen);
        if (client_sock < 0) {
            perror("accept");
            continue;
        }
        std::cout << "클라이언트 연결됨\n";
        handle_client(client_sock);
        close(client_sock);
    }

    return 0;
}
*/
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#define PORT 9888
#define BUFFER_SIZE 65536

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
Ort::Session* session = nullptr;
Ort::SessionOptions session_options;

const int INPUT_W = 640;
const int INPUT_H = 640;

std::vector<const char*> input_node_names = {"images"};
std::vector<const char*> output_node_names = {"output0"};  // YOLOv8 기준

void initialize_model(const std::string& model_path) {
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = new Ort::Session(env, model_path.c_str(), session_options);
}

std::vector<float> run_onnx_inference(const cv::Mat& input_img) {
    cv::Mat resized;
    cv::resize(input_img, resized, cv::Size(INPUT_W, INPUT_H));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // [1, 3, H, W]
    std::vector<float> input_tensor_values(1 * 3 * INPUT_H * INPUT_W);
    size_t idx = 0;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < INPUT_H; ++y) {
            for (int x = 0; x < INPUT_W; ++x) {
                input_tensor_values[idx++] = resized.at<cv::Vec3f>(y, x)[c];
            }
        }
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, 3, INPUT_H, INPUT_W};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    // 추론
    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
        input_node_names.data(), &input_tensor, 1,
        output_node_names.data(), 1);

    float* output = output_tensors[0].GetTensorMutableData<float>();
    size_t num_outputs = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    // 여기서는 첫 번째 바운딩박스 [x1, y1, x2, y2]만 반환
    std::vector<float> bbox;
    if (num_outputs >= 6) {
        bbox.assign(output, output + 4);  // x1, y1, x2, y2
    } else {
        bbox = {0, 0, 0, 0};  // 실패 대비
    }

    return bbox;
}

void handle_client(int client_sock) {
    char buffer[BUFFER_SIZE];
    int bytes_received = recv(client_sock, buffer, BUFFER_SIZE, 0);
    if (bytes_received <= 0) {
        std::cerr << "클라이언트로부터 데이터를 받지 못함\n";
        return;
    }

    std::vector<uchar> img_data(buffer, buffer + bytes_received);
    cv::Mat img = cv::imdecode(img_data, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "이미지 디코딩 실패\n";
        return;
    }

    std::vector<float> bbox = run_onnx_inference(img);

    // 바운딩박스 전송
    send(client_sock, bbox.data(), bbox.size() * sizeof(float), 0);
}

int main() {
    initialize_model("./models/weights14/best.onnx");  // 모델 로드

    int server_fd, client_sock;
    struct sockaddr_in address;
    socklen_t addrlen = sizeof(address);

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    bind(server_fd, (struct sockaddr*)&address, sizeof(address));
    listen(server_fd, 3);
    std::cout << "서버 대기 중...\n";

    while (true) {
        client_sock = accept(server_fd, (struct sockaddr*)&address, &addrlen);
        if (client_sock < 0) {
            perror("accept");
            continue;
        }
        std::cout << "클라이언트 연결됨\n";
        handle_client(client_sock);
        close(client_sock);
    }

    delete session;
    return 0;
}