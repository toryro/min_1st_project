#include <iostream>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <numeric>

#define PORT 9888

// --- 추론 관련 설정 ---
//const std::string MODEL_PATH = "/Users/tory/Tory/02.Study/01.1team/min_1st_project/models/yolov8n.onnx"; // 사용할 ONNX 모델 경로
const std::string MODEL_PATH = "/Users/tory/Tory/02.Study/01.1team/min_1st_project/models/weights14/best.onnx"; // 사용할 ONNX 모델 경로

const float CONFIDENCE_THRESHOLD = 0.15;        // 최소 신뢰도 기본 0.3
const float NMS_THRESHOLD = 0.25;               // NMS 임계값 기본 0.25

/**
 * @brief ONNX 추론을 담당하는 헬퍼 클래스
 */
class InferenceHelper {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<int64_t> input_dims;
    std::vector<int64_t> output_dims;
    std::vector<std::string> input_names_str; // <<< 여기 수정
    std::vector<std::string> output_names_str; // <<< 여기 수정

    int input_h;
    int input_w;

public:
    InferenceHelper() : env(ORT_LOGGING_LEVEL_WARNING, "ONNX_SERVER"), session(nullptr) {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        try {
            session = Ort::Session(env, MODEL_PATH.c_str(), session_options);

            size_t num_input_nodes = session.GetInputCount();
            size_t num_output_nodes = session.GetOutputCount();

            // 입력 노드 정보
            input_names_str.resize(num_input_nodes); // <<< resize
            for (size_t i = 0; i < num_input_nodes; i++) {
                // GetInputNameAllocated는 Ort::AllocatedStringPtr을 반환
                // 이는 자동으로 소멸될 때 메모리를 해제합니다.
                // 따라서 Get()으로 얻은 char*를 std::string으로 복사해야 합니다.
                auto input_name_allocated = session.GetInputNameAllocated(i, allocator);
                input_names_str[i] = input_name_allocated.get(); // <<< 문자열 복사
                Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                input_dims = tensor_info.GetShape();
            }

            // 출력 노드 정보
            output_names_str.resize(num_output_nodes); // <<< resize
            for (size_t i = 0; i < num_output_nodes; i++) {
                auto output_name_allocated = session.GetOutputNameAllocated(i, allocator);
                output_names_str[i] = output_name_allocated.get(); // <<< 문자열 복사
                Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                output_dims = tensor_info.GetShape();
            }
        } catch (const Ort::Exception& e) {
            std::cerr << "모델 로드 중 오류 발생: " << e.what() << std::endl;
            std::cerr << "ONNX Runtime 에러 코드: " << std::endl;
            // 로드 실패 시 프로그램 종료 (예시)
            exit(EXIT_FAILURE); 
        }
        
        input_h = input_dims[2];
        input_w = input_dims[3];

        std::cout << "ONNX 모델 로드 완료: " << MODEL_PATH << std::endl;
        std::cout << "입력 차원 (NCHW): " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;

    }

    // 추론 실행 및 결과 문자열 반환
    std::string run_inference(const cv::Mat& original_image) {
        cv::Mat processed_image;
        float scale;
        cv::Mat preprocessed_blob = preprocess(original_image, processed_image, scale);

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, preprocessed_blob.ptr<float>(), preprocessed_blob.total(), input_dims.data(), input_dims.size());
        
        // input_names와 output_names를 const char* 배열로 변환
        std::vector<const char*> input_names_c_str;
        for (const auto& name : input_names_str) {
            input_names_c_str.push_back(name.c_str());
        }

        std::vector<const char*> output_names_c_str;
        for (const auto& name : output_names_str) {
            output_names_c_str.push_back(name.c_str());
        }

        // Run 함수 호출 시 변환된 const char* 배열 사용
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_c_str.data(), &input_tensor, 1, output_names_c_str.data(), 1);

        return postprocess(output_tensors[0], scale, original_image.size());
    }

private:
    // 전처리 함수
    cv::Mat preprocess(const cv::Mat& image, cv::Mat& out_image, float& out_scale) {
        // 레터박싱으로 비율을 유지하며 리사이즈
        float r = std::min((float)input_w / image.cols, (float)input_h / image.rows);
        out_scale = r;
        int new_unpad_w = r * image.cols;
        int new_unpad_h = r * image.rows;
        
        cv::resize(image, out_image, cv::Size(new_unpad_w, new_unpad_h));

        // 패딩 추가
        int top = (input_h - new_unpad_h) / 2;
        int bottom = input_h - new_unpad_h - top;
        int left = (input_w - new_unpad_w) / 2;
        int right = input_w - new_unpad_w - left;
        cv::copyMakeBorder(out_image, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        // BGR to RGB, uint8 to float32, Normalize
        cv::Mat blob;
        cv::cvtColor(out_image, blob, cv::COLOR_BGR2RGB);
        blob.convertTo(blob, CV_32F, 1.0/255.0);

        // HWC to CHW
        return cv::dnn::blobFromImage(blob);
    }

    // 후처리 함수
    std::string postprocess(Ort::Value& output_tensor, float scale, const cv::Size& original_img_size) {
        float* raw_output = output_tensor.GetTensorMutableData<float>();
        
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;

        // YOLOv8 출력 형식 [1, 84, 8400]을 가정 (x_center, y_center, w, h, class_probs...)
        // 모델에 따라 이 부분의 구조가 달라질 수 있습니다.
        cv::Mat output_mat(output_dims[1], output_dims[2], CV_32F, raw_output);
        output_mat = output_mat.t(); // Transpose to [8400, 84]

        for (int i = 0; i < output_mat.rows; i++) {
            float confidence = output_mat.at<float>(i, 4); // 예시: 5번째 값이 전체 confidence
            if (confidence > CONFIDENCE_THRESHOLD) {
                float cx = output_mat.at<float>(i, 0);
                float cy = output_mat.at<float>(i, 1);
                float w = output_mat.at<float>(i, 2);
                float h = output_mat.at<float>(i, 3);

                // 좌표 스케일링
                int left = static_cast<int>((cx - w / 2 - (input_w - original_img_size.width * scale) / 2) / scale);
                int top = static_cast<int>((cy - h / 2 - (input_h - original_img_size.height * scale) / 2) / scale);
                int width = static_cast<int>(w / scale);
                int height = static_cast<int>(h / scale);

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(confidence);
            }
        }

        // NMS (비최대 억제)
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

        // 결과 문자열 생성
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            const cv::Rect& rect = boxes[idx];
            ss << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height;
            if (i < indices.size() - 1) {
                ss << ", ";
            }
        }
        ss << "]";
        return ss.str();
    }
};

// --- Main 함수 및 소켓 통신 (이전과 동일) ---
int main() {
    InferenceHelper inference_helper;

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
        uint32_t data_size;
        int bytes_read = read(new_socket, &data_size, sizeof(data_size));
        if (bytes_read <= 0) {
            std::cout << "클라이언트 연결 끊김." << std::endl;
            break;
        }
        data_size = ntohl(data_size);

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
        
        cv::Mat image = cv::imdecode(img_buffer, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "이미지 디코딩 실패" << std::endl;
            continue;
        }

        // 추론 실행
        std::string bounding_boxes_str = inference_helper.run_inference(image);

        send(new_socket, bounding_boxes_str.c_str(), bounding_boxes_str.length(), 0);
        std::cout << "결과 전송: " << bounding_boxes_str << std::endl;
    }

    close(new_socket);
    close(server_fd);

    return 0;
}