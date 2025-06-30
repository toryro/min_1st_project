// server.cpp
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// 안전 수신: 정확히 len 바이트를 받을 때까지 반복
bool readN(int fd, void* buf, size_t len) {
    size_t recvd = 0;
    while (recvd < len) {
        ssize_t n = recv(fd, (char*)buf + recvd, len - recvd, 0);
        if (n <= 0) return false;

        std::cout << "buf len : " << n << "\n";
        recvd += n;
    }
    return true;
}
// 안전 송신
bool writeN(int fd, const void* buf, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, (const char*)buf + sent, len - sent, 0);
        if (n <= 0) return false;
        sent += n;
    }
    return true;
}

int main(int argc, char** argv) {
    const int PORT = 9888;
    srand((unsigned)time(nullptr));

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(PORT);
    bind(srv, (sockaddr*)&addr, sizeof(addr));
    listen(srv, 1);
    std::cout << "🚀  Listening on " << PORT << '\n';

    int cli = accept(srv, nullptr, nullptr);
    std::cout << "✅  Client connected\n";

    while (true) {
        uint32_t len_net;
        if (!readN(cli, &len_net, 4)) {
            std::cout << "❌  연결 종료 (1)\n";
            break;
        }
        uint32_t len = ntohl(len_net);
        std::vector<uchar> buf(len);
        if (!readN(cli, buf.data(), len)) {
            std::cout << "❌  연결 종료 (2)\n";
            break;
        }

        // 디코딩 (필수는 아님 – 그냥 크기 확인용)
        cv::Mat frame = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (frame.empty()) {
            std::cout << "❌  연결 종료 (3)\n";
            break;
        }
        int w = frame.cols, h = frame.rows;

        // 무작위 bbox 생성
        int x1 = rand() % (w / 2);
        int y1 = rand() % (h / 2);
        int x2 = x1 + rand() % (w / 2);
        int y2 = y1 + rand() % (h / 2);

        //int32_t bbox[4] = { htonl(x1), htonl(y1), htonl(x2), htonl(y2) };
        int32_t bbox[4];
        bbox[0] = htonl(x1);
        bbox[1] = htonl(y1);
        bbox[2] = htonl(x2);
        bbox[3] = htonl(y2);
        if (!writeN(cli, bbox, sizeof(bbox))) {
            std::cout << "❌  연결 종료 (4)\n";
            break;
        }
    }
    close(cli); close(srv);
    std::cout << "❌  연결 종료\n";
    return 0;
}