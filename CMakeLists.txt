cmake_minimum_required(VERSION 3.10)
project(OnnxMNIST)

# ✅ C++17 사용 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ✅ ONNX Runtime 경로 설정
include_directories($ENV{HOME}/onnxruntime/include)
link_directories($ENV{HOME}/onnxruntime/lib)

# ✅ OpenCV 찾기
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ✅ 실행 파일 생성 및 라이브러리 연결
add_executable(mnist_infer infer.cpp)
target_link_libraries(mnist_infer ${OpenCV_LIBS} onnxruntime stdc++fs)  # filesystem 링크
