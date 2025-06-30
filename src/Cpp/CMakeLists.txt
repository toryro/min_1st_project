cmake_minimum_required(VERSION 3.10)
project(MyOpenCVProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OpenCV 찾기
find_package(OpenCV REQUIRED)

# include 디렉토리 설정
include_directories(${OpenCV_INCLUDE_DIRS})

# 실행 파일 설정
add_executable(draw_server draw_server.cpp)

# OpenCV 라이브러리 링크
target_link_libraries(draw_server ${OpenCV_LIBS})