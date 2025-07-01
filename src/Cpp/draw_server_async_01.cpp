// draw_server_async_fixed.cpp
// 빌드: g++ -std=c++17 draw_server_async_fixed.cpp `pkg-config --cflags --libs opencv4` -lonnxruntime -lpthread -o server
// 실행: ./server 0.0.0.0 9888 yolov8n.onnx
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
constexpr int   NUM_CLASSES = 80;
//constexpr float CONF_THR = 0.35f, NMS_THR = 0.45f;
constexpr float CONF_THR = 0.35f, NMS_THR = 0.45f;

/* ───── 전처리 (호출자 소유 blob) ───────────────────────────────────────── */
Ort::Value preprocess(const cv::Mat& src,
                      std::vector<float>& blob,
                      float& scale,
                      Ort::MemoryInfo& mem)
{
    int w = src.cols, h = src.rows;
    scale = std::min(INPUT_W/(float)w, INPUT_H/(float)h);
    int nw = int(w * scale), nh = int(h * scale);

    cv::Mat resized;  cv::resize(src, resized, {nw, nh});
    cv::Mat canvas(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114,114,114));
    resized.copyTo(canvas(cv::Rect(0,0,nw,nh)));

    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
    canvas.convertTo(canvas, CV_32F, 1.0/255.0);

    std::vector<cv::Mat> ch(3); cv::split(canvas, ch);
    for (int i = 0; i < 3; ++i)
        std::memcpy(blob.data()+i*INPUT_H*INPUT_W, ch[i].data, INPUT_H*INPUT_W*sizeof(float));

    std::vector<int64_t> dims{1,3,INPUT_H,INPUT_W};
    return Ort::Value::CreateTensor<float>(mem, blob.data(), blob.size(), dims.data(), dims.size());
}

/* ───── 후처리 (NMS + 클래스라벨) ─────────────────────────────────────── */
std::string postprocess(const Ort::Value& out,
                        float scale, const cv::Size&)
{
    const float* p = out.GetTensorData<float>();
    auto shp = out.GetTensorTypeAndShapeInfo().GetShape();   // [1,84,8400]
    int N = (int)shp[2];

    std::vector<cv::Rect> boxes; std::vector<float> scores; std::vector<int> cls;
    auto sig = [](float x){ return 1.f / (1.f + std::exp(-x)); };
    for (int i = 0; i < N; ++i) {
        
        float obj = sig(p[4*N + i]);
        if (obj < CONF_THR) continue;

        /* ── 클래스 선택 ────────────────────── */
        float best_cls = 0.f; int best_id = -1;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float cls = sig(p[(5 + c)*N + i]);
            if (cls > best_cls) { best_cls = cls; best_id = c; }
        }

        float conf = obj * best_cls;
        if (conf < CONF_THR) continue;              // ① 점수 컷

        /* ── BBox 좌표 복원 ─────────────────── */
        float cx = p[0*N + i], cy = p[1*N + i];
        float bw = p[2*N + i], bh = p[3*N + i];
        float x1 = (cx - bw/2.f) / scale;
        float y1 = (cy - bh/2.f) / scale;

        /* ── ② 면적 필터 ────────────────────── */
        float area = (bw * bh) / (INPUT_W * INPUT_H);   // 상대 면적 (0~1)
        if (area < 0.0005f) // 0.05 % 미만은 스킵
            continue;

        /* ── 통과 ⇒ push_back ──────────────── */
        boxes.emplace_back(cv::Rect(round(x1), round(y1), round(bw/scale), round(bh/scale)));
        scores.emplace_back(conf);
        cls.emplace_back(best_id);

        // float obj = sig(p[4*N + i]);
        // //float obj = p[4*N + i];
        
        // if (obj < CONF_THR) continue;
        // float best = 0.f; int best_id = -1;
        // for (int c = 0; c < NUM_CLASSES; ++c) {

        //     float v = sig(p[(5 + c)*N + i]);
        //     //float v = p[(5 + c)*N + i];
        //     if (v > best) { best = v; best_id = c; }
        // }
        // float conf = obj * best; if (conf < CONF_THR) continue;

        // float cx=p[0*N+i], cy=p[1*N+i], w=p[2*N+i], h=p[3*N+i];
        // float x1=(cx-w/2)/scale, y1=(cy-h/2)/scale;
        // boxes.emplace_back(cv::Rect(round(x1),round(y1),round(w/scale),round(h/scale)));
        // scores.push_back(conf); cls.push_back(best_id);
        // NMS 전, 상위 20개 로그 출력 (디버깅용)
        //if (i < 20)
            std::cout << "pred " << i << " cls=" << best_id << " conf=" << conf << '\n';
    }
    std::vector<int> keep; cv::dnn::NMSBoxes(boxes,scores,CONF_THR,NMS_THR,keep);

    std::ostringstream ss; ss<<'[';
    for(size_t k=0;k<keep.size();++k){
        int i=keep[k]; const auto& r=boxes[i];
        ss<<r.x<<','<<r.y<<','<<r.width<<','<<r.height<<','<<cls[i];
        if(k+1<keep.size()) ss<<',';
    }
    ss<<']'; return ss.str();
}

/* ───── TCP 헬퍼 ──────────────────────────────────────────────────────── */
bool recvAll(int s, void* b, size_t l){char* p=(char*)b;while(l){ssize_t n=recv(s,p,l,0);if(n<=0)return false;p+=n;l-=n;}return true;}
bool sendAll(int s,const void* b,size_t l){const char* p=(const char*)b;while(l){ssize_t n=send(s,p,l,0);if(n<=0)return false;p+=n;l-=n;}return true;}

int main(int argc,char* argv[])
{
    if(argc<4){std::cerr<<"Usage: "<<argv[0]<<" <bind_ip> <port> <model.onnx>\n";return 1;}
    const char* BIND_IP=argv[1]; int PORT=std::stoi(argv[2]); const char* MODEL=argv[3];

    std::cout<<"🔵 BIND_IP : "<<BIND_IP<<'\n';
    std::cout<<"🔵 PORT : " << PORT << '\n';
    std::cout<<"🔵 MODEL : " << MODEL << '\n';

    /* ── ORT 세션 ── */
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"srv");
    Ort::SessionOptions so; so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, MODEL, so);
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator,OrtMemTypeCPU);

    /* ── 입력·출력 이름 (수명 보장) ── */
    Ort::AllocatorWithDefaultOptions alloc;
    static std::vector<std::string>  in_strs,  out_strs;
    static std::vector<const char*>  in_names, out_names;

#if defined(ORT_API_VERSION) && ORT_API_VERSION >= 18
    in_strs  = session.GetInputNames();
    out_strs = session.GetOutputNames();
#else
    {   size_t n_in=session.GetInputCount(), n_out=session.GetOutputCount();
        in_strs.reserve(n_in);  out_strs.reserve(n_out);
        for(size_t i=0;i<n_in;++i)
            in_strs.emplace_back(session.GetInputName(i, alloc));
        for(size_t i=0;i<n_out;++i)
            out_strs.emplace_back(session.GetOutputName(i, alloc));
    }
#endif
    in_names.reserve(in_strs.size());  out_names.reserve(out_strs.size());
    for(auto& s: in_strs)  in_names.push_back(s.c_str());
    for(auto& s: out_strs) out_names.push_back(s.c_str());

    /* ── TCP 서버 ── */
    int srv=socket(AF_INET,SOCK_STREAM,0);
    sockaddr_in addr{}; addr.sin_family=AF_INET; addr.sin_port=htons(PORT);
    inet_pton(AF_INET,BIND_IP,&addr.sin_addr);
    int yes=1; setsockopt(srv,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
    if(bind(srv,(sockaddr*)&addr,sizeof(addr))<0||listen(srv,1)<0){perror("socket");return 1;}
    std::cout<<"🔵 Listening on "<<BIND_IP<<':'<<PORT<<'\n';

    while(true){
        int cli=accept(srv,nullptr,nullptr); if(cli<0)continue;
        std::cout<<"🟢 Client connected\n";

        std::vector<float> blob(INPUT_W*INPUT_H*3);

        while(true){
            uint32_t len_be; if(!recvAll(cli,&len_be,4)) break;
            uint32_t n=ntohl(len_be); std::vector<uchar> buf(n);
            if(!recvAll(cli,buf.data(),n)) break;

            cv::Mat img=cv::imdecode(buf,cv::IMREAD_COLOR); if(img.empty()) continue;

            float scale; Ort::Value input=preprocess(img,blob,scale,mem);

            std::vector<Ort::Value> outs;
            try{
                outs=session.Run(Ort::RunOptions{nullptr},
                                 in_names.data(), &input, in_names.size(),
                                 out_names.data(), out_names.size());
            }catch(const Ort::Exception& e){
                std::cerr<<"Run() failed: "<<e.what()<<'\n'; break;
            }

            std::string payload=postprocess(outs[0],scale,img.size());
            payload.push_back('\n');
            if(!sendAll(cli,payload.data(),payload.size())) break;
        }
        std::cout<<"🔴 Client disconnected\n";
        close(cli);
    }
    close(srv); return 0;
}