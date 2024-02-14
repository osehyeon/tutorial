#include <iostream>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>

int main() {
    std::cout << "MNIST Classifier with TensorFlow Lite" << std::endl;

    // Load TensorFlow Lite model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("./model/mnist12.tflite");

    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    if (!interpreter) {
        std::cerr << "Failed to construct interpreter" << std::endl;
        return 1;
    }

    std::cout << "TensorFlow Lite model loaded successfully" << std::endl;
    return 0;
}
