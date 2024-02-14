cmake_minimum_required(VERSION 3.16)
project(MNIST_Classifier)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# TensorFlow Lite paths
# Replace these with the actual paths
set(TENSORFLOW_LITE_LIB_PATH "${CMAKE_SOURCE_DIR}/tensorflow_src/bazel-bin/tensorflow/lite")
set(TENSORFLOW_LITE_INCLUDE_PATH "${CMAKE_SOURCE_DIR}/tensorflow_src")
set(FLATBUFFERS_INCLUDE_PATH "${CMAKE_SOURCE_DIR}/tensorflow_src/bazel-tensorflow_src/external/flatbuffers/include")

# Include directories
include_directories(${TENSORFLOW_LITE_INCLUDE_PATH})
include_directories(${FLATBUFFERS_INCLUDE_PATH})

add_library(libtensorflowlite SHARED IMPORTED)
set_target_properties(libtensorflowlite PROPERTIES IMPORTED_LOCATION "${TENSORFLOW_LITE_LIB_PATH}/libtensorflowlite.dylib")

add_subdirectory(src)
