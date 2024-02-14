# app/CMakeLists.txt
add_executable(mainapp main.cpp)

# libstatic과 libshared의 include 디렉토리를 절대 경로로 포함
target_include_directories(mainapp PRIVATE ${CMAKE_SOURCE_DIR}/libstatic ${CMAKE_SOURCE_DIR}/libshared)

# staticlib과 sharedlib 라이브러리와 연결
target_link_libraries(mainapp staticlib sharedlib)