cmake_minimum_required(VERSION 3.5)
project(pthreadpool_example C)

# 실행 파일을 생성하는데 필요한 소스 파일을 추가합니다.
set(SOURCES src/addition.c)

# 실행 파일 생성
add_executable(pthreadpool_example ${SOURCES})

# pthreadpool 라이브러리 포함 경로 추가
target_include_directories(pthreadpool_example PRIVATE include)

# pthreadpool 라이브러리 링크
target_link_libraries(pthreadpool_example PRIVATE ${CMAKE_SOURCE_DIR}/lib/libpthreadpool.a)
