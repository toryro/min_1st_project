// draw_server_async_auto.cpp  ── 최종판
// -----------------------------------------------------------------------------
// 빌드(예, macOS Homebrew):
//   brew install opencv onnxruntime
//   g++ -std=c++17 draw_server_async_auto.cpp \
//       `pkg-config --cflags --libs opencv4` -lonnxruntime -lpthread -o server
//
// 실행:
//   ./server 0.0.0.0 9888 yolov8n.onnx
// -----------------------------------------------------------------------------
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

constexpr int   INPUT_W = 640, INPUT_H = 640;
constexpr int   NUM_CLASSES = 80;              // COCO
constexpr float CONF_THR = 0.25f;
constexpr float NMS_THR  = 0.45f;

/* ───── 전처리 (호출자 소유 blob) ─────────────────────────────────────────── */
Ort::Value preprocess(const cv::Mat& src,
                      std::vector<float>& blob,
                      float& scale,
                      Ort::MemoryInfo& mem)
{
    int w = src.cols, h = src.rows;
    scale = std::min(INPUT_W / (float)w, INPUT_H / (float)h);
    int nw = int(w * scale), nh = int(h * scale);

    cv::Mat resized; cv::resize(src, resized, {nw, nh});
    cv::Mat canvas(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114,114,114));
    resized.copyTo(canvas(cv::Rect(0, 0, nw, nh)));

    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
    canvas.convertTo(canvas, CV_32F, 1.0/255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(canvas, channels);
    for (int c = 0; c < 3; ++c)
        std::memcpy(blob.data() + c*INPUT_H*INPUT_W,
                    channels[c].data,
                    INPUT_H*INPUT_W*sizeof(float));

    std::vector<int64_t> dims{1, 3, INPUT_H, INPUT_W};
    return Ort::Value::CreateTensor<float>(mem,
                                           blob.data(), blob.size(),
                                           dims.data(), dims.size());
}

/* ───── 후처리 (NMS + 클래스 ID 포함) ─────────────────────────────────────── */
std::string postprocess(const Ort::Value& out,
                        float scale,
                        const cv::Size& orig)
{
    const float* p = out.GetTensorData<float>();
    auto shape = out.GetTensorTypeAndShapeInfo().GetShape();   // [1,84,8400]
    int N = (int)shape[2];

    std::vector<cv::Rect> boxes;
    std::vector<float>    scores;
    std::vector<int>      cls_ids;

    for (int i = 0; i < N; ++i) {
        float obj = p[4*N + i];
        if (obj < CONF_THR) continue;

        float best_cls = 0.f; int best_id = -1;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float cls = p[(5+c)*N + i];
            if (cls > best_cls) { best_cls = cls; best_id = c; }
        }
        float conf = obj * best_cls;
        if (conf < CONF_THR) continue;

        float cx = p[0*N + i], cy = p[1*N + i];
        float w  = p[2*N + i], h  = p[3*N + i];

        float x1 = (cx - w/2.f) / scale;
        float y1 = (cy - h/2.f) / scale;

        boxes.emplace_back(cv::Rect(round(x1), round(y1),
                                    round(w/scale), round(h/scale)));
        scores.emplace_back(conf);
        cls_ids.emplace_back(best_id);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, CONF_THR, NMS_THR, keep);

    std::ostringstream ss; ss << '[';
    for (size_t k = 0; k < keep.size(); ++k) {
        int i = keep[k];
        const auto& r = boxes[i];
        ss << r.x << ',' << r.y << ',' << r.width << ',' << r.height << ','
           << cls_ids[i];
        if (k + 1 < keep.size()) ss << ',';
    }
    ss << ']';
    return ss.str();
}

/* ───── TCP 헬퍼 ─────────────────────────────────────────────────────────── */
bool recvAll(int s, void* buf, size_t len) {
    char* p = (char*)buf;
    while (len) { ssize_t n = recv(s, p, len, 0); if (n <= 0) return false;
                  p += n; len -= n; }
    return true;
}
bool sendAll(int s, const void* buf, size_t len) {
    const char* p = (const char*)buf;
    while (len) { ssize_t n = send(s, p, len, 0); if (n <= 0) return false;
                  p += n; len -= n; }
    return true;
}

/* ───── 메인 ─────────────────────────────────────────────────────────────── */
int main(int argc, char* argv[])
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <bind_ip> <port> <model.onnx>\n";
        return 1;
    }
    const char* BIND_IP = argv[1];
    int         PORT    = std::stoi(argv[2]);
    const char* MODEL   = argv[3];

    /* ONNX Runtime 초기화 */
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "srv");
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, MODEL, so);
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
                              OrtDeviceAllocator, OrtMemTypeCPU);

    /* ─────────── 입·출력 이름 안전하게 얻기 ─────────── */
    Ort::AllocatorWithDefaultOptions alloc;
    size_t in_cnt  = session.GetInputCount();
    size_t out_cnt = session.GetOutputCount();

    std::vector<Ort::AllocatedStringPtr> in_buf, out_buf;
    std::vector<const char*> in_names, out_names;

    for (size_t i = 0; i < in_cnt;  ++i) {
        in_buf.emplace_back(session.GetInputNameAllocated(i, alloc));
        in_names.push_back(in_buf.back().get());
    }
    for (size_t i = 0; i < out_cnt; ++i) {
        out_buf.emplace_back(session.GetOutputNameAllocated(i, alloc));
        out_names.push_back(out_buf.back().get());
    }

    /* ── TCP 서버 설정 ── */
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{}; addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    inet_pton(AF_INET, BIND_IP, &addr.sin_addr);
    int yes = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    if (bind(srv, (sockaddr*)&addr, sizeof(addr)) < 0 || listen(srv, 1) < 0) {
        perror("socket"); return 1;
    }
    std::cout << "🔵 Listening on " << BIND_IP << ':' << PORT << '\n';

    while (true) {
        int cli = accept(srv, nullptr, nullptr);
        if (cli < 0) continue;
        std::cout << "🟢 Client connected\n";

        std::vector<float> blob(INPUT_W * INPUT_H * 3);  // 입력 버퍼

        while (true) {
            uint32_t len_be;
            if (!recvAll(cli, &len_be, 4)) break;
            uint32_t jlen = ntohl(len_be);
            std::vector<uchar> jpeg(jlen);
            if (!recvAll(cli, jpeg.data(), jlen)) break;

            cv::Mat img = cv::imdecode(jpeg, cv::IMREAD_COLOR);
            if (img.empty()) continue;

            float scale;
            Ort::Value input = preprocess(img, blob, scale, mem);

            std::vector<Ort::Value> outputs;
            try {
                outputs = session.Run(Ort::RunOptions{nullptr},
                                      in_names.data(),  &input, 1,
                                      out_names.data(), 1);
            } catch (const Ort::Exception& e) {
                std::cerr << "Run() failed: " << e.what() << '\n';
                break;
            }

            std::string payload = postprocess(outputs[0], scale, img.size());
            payload.push_back('\n');
            if (!sendAll(cli, payload.data(), payload.size())) break;
        }
        std::cout << "🔴 Client disconnected\n";
        close(cli);
    }
    close(srv);
    return 0;
}