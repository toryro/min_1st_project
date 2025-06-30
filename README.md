# min_1st_project

# 25.06.30 - 001
# 프로젝트 폴더 구조 변경
# C++에서 ONNX + ONNXRuntime 를 사용하기위해 onnxruntime를 추가하여야 함.
# onnxruntime는 https://github.com/microsoft/onnxruntime/releases 에서 1.22.0을 받음.
min_1st_project/
├── models/
│   └── weightsXX/
│       ├── best.pt
│       ├── last.pt
│       └── best.onnx
├── src/
│   └── Cpp
│   │   └── test_server.cpp
│   └── python
│       └── test_client.py
├── onnxruntime-install/
│   ├── include/
│   │   └── onnxruntime_cxx_api.h
│   └── lib/
│       └── libonnxruntime.dylib
└── CMakeLists.txt (또는 .vscode 폴더)
# test.ipynb에 best.pt를 best.onnx로 변환하는 방법 있음.
# CMakeList.txt는 좀더 알아봐야함.(사용법에 관해서)

# 25.06.30 - 002
# draw_server_G_01.cpp, draw_client_G_01.py
# 영상 출력하고, 영상 보내고, 서버에서 추론 후 좌표 보내고, 클라이언트에서 바운딩박스 그려주는것까지 확인했음.