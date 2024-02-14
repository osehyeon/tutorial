# Add the executable
add_executable(mnist_classifier mnist.cpp)
add_executable(system_setting main.cpp)

# Link TensorFlow Lite
target_link_libraries(mnist_classifier libtensorflowlite)
target_link_libraries(system_setting libtensorflowlite)

add_custom_command(
    TARGET mnist_classifier POST_BUILD
    COMMAND python3 ${CMAKE_HOME_DIRECTORY}/python/1_mnist_model_gen.py &&
            python3 ${CMAKE_HOME_DIRECTORY}/python/2_mnist_dataset_gen.py
)
