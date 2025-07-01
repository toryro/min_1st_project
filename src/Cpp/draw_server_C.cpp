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

const int INPUT_W = 640;
const int INPUT_H = 640;
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_server");
Ort::Session* session = nullptr;
Ort::SessionOptions session_options;

std::vector<const char*> input_node_names;
std::vector<const char*> output_node_names;

void initialize_model(const std::string& model_path) {
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = new Ort::Session(env, model_path.c_str(), session_options);

    // 입력/출력 노드 이름 얻기
    std::vector<std::string> input_names = session->GetInputNames();
    for (const auto& name : input_names)
        input_node_names.push_back(name.c_str());

    std::vector<std::string> output_names = session->GetOutputNames();
    for (const auto& name : output_names)
        output_node_names.push_back(name.c_str());
}

std::vector<float> run_onnx_inference(const cv::Mat& input_img) {
    cv::Mat resized;
    cv::resize(input_img, resized, cv::Size(INPUT_W, INPUT_H));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // [1, 3, H, W]로 텐서 준비
    std::vector<float> input_tensor_values(1 * 3 * INPUT_H * INPUT_W);
    size_t idx = 0;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < INPUT_H; ++y) {
            for (int x = 0; x < INPUT_W; ++x) {
                input_tensor_values[idx++] = resized.at<cv::Vec3f>(y, x)[c];
            }
        }
    }

    std::vector<int64_t> input_shape = {1, 3, INPUT_H, INPUT_W};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
        input_node_names.data(), &input_tensor, 1,
        output_node_names.data(), 1);

    float* output = output_tensors[0].GetTensorMutableData<float>();
    size_t num_outputs = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    // [x1, y1, x2, y2, conf, cls] 기준 → 첫 번째 바운딩박스만 추출
    std::vector<float> bbox(4, 0.0f);
    if (num_outputs >= 6)
        std::copy(output, output + 4, bbox.begin());

    return bbox;
}

void handle_client(int client_sock) {
    char buffer[BUFFER_SIZE];
    int bytes_received = recv(client_sock, buffer, BUFFER_SIZE, 0);
    if (bytes_received <= 0) {
        std::cerr << "❌ 클라이언트로부터 데이터를 받지 못했습니다.\n";
        return;
    }

    std::vector<uchar> img_data(buffer, buffer + bytes_received);
    cv::Mat img = cv::imdecode(img_data, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "❌ 이미지 디코딩 실패\n";
        return;
    }

    std::vector<float> bbox = run_onnx_inference(img);
    send(client_sock, bbox.data(), bbox.size() * sizeof(float), 0);
}

int main() {
    initialize_model("/Users/tory/Tory/02.Study/01.1team/min_1st_project/models/weights14/best.onnx");  // 모델 파일 경로

    int server_fd, client_sock;
    struct sockaddr_in address;
    socklen_t addrlen = sizeof(address);

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("❌ 소켓 생성 실패");
        return -1;
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("❌ 바인딩 실패");
        return -1;
    }

    listen(server_fd, 3);
    std::cout << "🚀 서버가 대기 중입니다...\n";

    while (true) {
        client_sock = accept(server_fd, (struct sockaddr*)&address, &addrlen);
        if (client_sock < 0) {
            perror("❌ 클라이언트 수락 실패");
            continue;
        }

        std::cout << "✅ 클라이언트 연결됨\n";
        handle_client(client_sock);
        close(client_sock);
    }

    delete session;
    return 0;
}