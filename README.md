# min_1st_project

# 25.06.30 - 001
# 프로젝트 폴더 구조 변경
# C++에서 ONNX + ONNXRuntime 를 사용하기위해 onnxruntime를 추가하여야 함.
# onnxruntime는 https://github.com/microsoft/onnxruntime/releases 에서 1.22.0을 받음.
min_1st_project/
├── models/
│   └── weightsXX/   # XX는 일련번호임.
│       ├── best.pt
│       ├── last.pt
│       └── best.onnx
├── src/
│   ├── Cpp
│   │   ├── test_server.cpp
│   │   └── draw_server_async.cpp
│   └── python
│       ├── test_client.py
│       └── draw_server_async.cpp
├── onnxruntime/    # onnxruntime-install/ => onnxruntime/ 로 변경함.
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

# 25.07.01 -- 003
# draw_server_async.cpp, draw_client_async.py 파일 추가함.
# 비동기 처리/멀티스레딩과 파이프라이닝 추가해서 동작 확인함.
# 속도 개선에 많은 도움을 예상했지만...... 미미한 효과만 있음. ㅠ.ㅠ

# 25.07.01 -- 004
# draw_server_async_01.cpp, draw_client_async_01.py, draq_config.json 파일 추가함.

# draw_server_async_01.cpp는 컴파일 후 실행방법은 
# ./draw_server_async_01.out 0.0.0.0 9888 best.onnx
#                        바인딩 주소   포트 번호   모델경로

# draq_config.json에는 서버 주소, 포트 번호, 비디오 소스가 들어있다.
# draw_client_async_01.py를 바로 실행시키면 된다.

# -- 003에 비해 개선된 속도는 많이 개선되었지만, 객체 검출에 문제가 있는것 같다.

# draw_server_async_01.cpp 파일은 임시로 올려두었으며 차후 삭제 예정이다.