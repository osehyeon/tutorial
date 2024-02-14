# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)  # CMake 최소 버전 지정
project(MyProject VERSION 1.0)       # 프로젝트 이름과 버전 설정

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 실행 파일 추가
add_executable(MyProject src/main.cpp)