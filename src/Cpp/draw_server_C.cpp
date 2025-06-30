#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

constexpr int PORT = 9888;
constexpr int INPUT_WH = 640;          // YOLOv8n.onnx 입력 크기
constexpr float CONF_THR = 0.4f;
constexpr float NMS_IOU = 0.5f;

// ── 유틸: 소켓에서 정확히 n바이트 읽기 ──────────────────────────────
bool recv_all(int fd, void* buf, size_t len) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t r = 0;
    while (r < len) {
        ssize_t n = recv(fd, p + r, len - r, 0);
        if (n <= 0) return false;  // 연결 종료·에러
        r += n;
    }
    return true;
}

// ── NMS (간단한 IoU 기반) ──────────────────────────────────────────
struct Detection {
    float x1, y1, x2, y2, conf;
    int   cls;
};
float iou(const Detection& a, const Detection& b) {
    const float xx1 = std::max(a.x1, b.x1), yy1 = std::max(a.y1, b.y1);
    const float xx2 = std::min(a.x2, b.x2), yy2 = std::min(a.y2, b.y2);
    const float w = std::max(0.f, xx2 - xx1), h = std::max(0.f, yy2 - yy1);
    const float inter = w * h;
    const float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (areaA + areaB - inter + 1e-6f);
}
std::vector<Detection> nms(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(),
              [](auto& a, auto& b) { return a.conf > b.conf; });
    std::vector<Detection> keep;
    std::vector<char> removed(dets.size(), 0);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        keep.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j)
            if (!removed[j] && iou(dets[i], dets[j]) > NMS_IOU) removed[j] = 1;
    }
    return keep;
}

// ── 메인 ───────────────────────────────────────────────────────────
int main() {
    // 1) ONNX Runtime 초기화
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "server");
    Ort::SessionOptions opt;
    opt.SetIntraOpNumThreads(4);
    Ort::Session session(env, "yolov8n.onnx", opt);
    Ort::AllocatorWithDefaultOptions allocator;
    const char* input_name  = session.GetInputName(0, allocator);
    const char* output_name = session.GetOutputName(0, allocator);

    // 2) 소켓 오픈
    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(PORT);
    addr.sin_addr.s_addr = INADDR_ANY;
    bind(sfd, (sockaddr*)&addr, sizeof(addr));
    listen(sfd, 1);
    std::cout << "🟢 Listening on port " << PORT << "\n";

    int cfd = accept(sfd, nullptr, nullptr);
    std::cout << "✅ Client connected\n";

    while (true) {
        // 3) 프레임 길이 수신
        uint32_t len_net;
        if (!recv_all(cfd, &len_net, 4)) break;
        uint32_t len = ntohl(len_net);
        std::vector<uint8_t> jpeg(len);
        if (!recv_all(cfd, jpeg.data(), len)) break;

        // 4) JPEG 디코딩
        cv::Mat img = cv::imdecode(jpeg, cv::IMREAD_COLOR);
        if (img.empty()) continue;

        // 5) 전처리 (BGR→RGB, 리사이즈, 0~1 정규화)
        cv::Mat resized;
        cv::resize(img, resized, {INPUT_WH, INPUT_WH});
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);

        // CHW 텐서 만들기
        std::array<int64_t, 4> shape{1, 3, INPUT_WH, INPUT_WH};
        std::vector<float> blob(3 * INPUT_WH * INPUT_WH);
        size_t idx = 0;
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < INPUT_WH; ++y)
                for (int x = 0; x < INPUT_WH; ++x)
                    blob[idx++] = resized.at<cv::Vec3f>(y, x)[c];

        // 6) 추론
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            allocator, blob.data(), blob.size(), shape.data(), shape.size());

        auto output = session.Run(Ort::RunOptions{nullptr},
                                  &input_name, &input_tensor, 1,
                                  &output_name, 1);

        const float* out = output[0].GetTensorData<float>();
        const auto dims = output[0].GetTensorTypeAndShapeInfo().GetShape();
        const int   num_det = dims[1];     // (1, N, …)
        const int   dim_per = dims[2];     // 6 = x,y,w,h,conf,class

        std::vector<Detection> detections;
        for (int i = 0; i < num_det; ++i) {
            const float conf = out[i * dim_per + 4];
            if (conf < CONF_THR) continue;
            Detection d;
            const float cx = out[i * dim_per + 0] * img.cols;
            const float cy = out[i * dim_per + 1] * img.rows;
            const float w  = out[i * dim_per + 2] * img.cols;
            const float h  = out[i * dim_per + 3] * img.rows;
            d.x1  = cx - w / 2;  d.y1 = cy - h / 2;
            d.x2  = cx + w / 2;  d.y2 = cy + h / 2;
            d.conf = conf;
            d.cls  = static_cast<int>(out[i * dim_per + 5]);
            detections.push_back(d);
        }
        detections = nms(detections);

        // 7) 좌표 문자열 작성 "x1,y1,x2,y2,cls,conf;..."
        std::ostringstream oss;
        for (auto& d : detections) {
            oss << int(d.x1) << ',' << int(d.y1) << ','
                << int(d.x2) << ',' << int(d.y2) << ','
                << d.cls << ',' << d.conf << ';';
        }
        oss << '\n';
        std::string msg = oss.str();
        send(cfd, msg.data(), msg.size(), 0);          // 클라이언트 전송
        std::cout << msg;                               // 서버 콘솔에도 출력
    }
    close(cfd); close(sfd);
}