#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>

typedef struct {
    float *image_data;
    unsigned char *label_data;
} MNISTData;

float *load_32bit_float_array_from_file(const char *filename, size_t *size);
uint8_t *load_8bit_uint_array_from_file(const char *filename, size_t *size);
int mnist_datasets(MNISTData *dataset, const char *image_path, const char *label_path, int num_images);

int main()
{
    const char* model_path = "./model/mnist12.tflite";
    const char* image_path = "./data/images";
    const char* label_path = "./data/labels";
    int num_images = 10000;
    MNISTData mnist_test[num_images];

    // TensorFlow Lite 모델 로드
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    // XNNPACK Delegate 생성 및 적용
    TfLiteXNNPackDelegateOptions xnnpack_options = TfLiteXNNPackDelegateOptionsDefault();
    xnnpack_options.num_threads = -1; // 시스템에 따라 스레드 수 자동 결정
    TfLiteDelegate* xnnpack_delegate = TfLiteXNNPackDelegateCreate(&xnnpack_options);
    interpreter->ModifyGraphWithDelegate(xnnpack_delegate);

    // 인터프리터에 텐서 메모리 할당
    interpreter->AllocateTensors();

    // 데이터 로드
    if (mnist_datasets(mnist_test, image_path, label_path, num_images) != 0) {
        fprintf(stderr, "Failed to load MNIST data\n");
        return 1;
    }

    // 추론 실행 및 정확도 계산
    int correct_predictions = 0;
    for (int i = 0; i < num_images; i++) {
        // 입력 텐서 설정
        float* input = interpreter->typed_input_tensor<float>(0);
        memcpy(input, mnist_test[i].image_data, 28 * 28 * sizeof(float));

        // 추론 실행
        interpreter->Invoke();

        // 출력 텐서에서 결과 가져오기
        float* output = interpreter->typed_output_tensor<float>(0);

        // 모델의 출력에서 예측된 라벨을 계산
        int predicted_label = 0;
        float max_score = output[0];
        for (int j = 1; j < 10; j++) {
            if (output[j] > max_score) {
                max_score = output[j];
                predicted_label = j;
            }
        }

        // 정확도 계산
        if (predicted_label == mnist_test[i].label_data[0]) {
            correct_predictions++;
        }
    }

    // 결과 출력
    printf("Accuracy: %d/%d\n", correct_predictions, num_images);

    // XNNPACK Delegate 해제
    TfLiteXNNPackDelegateDelete(xnnpack_delegate);
    return 0;
}


float *load_32bit_float_array_from_file(const char *filename, size_t *size)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        perror("Error opening file");
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    *size = file_size / sizeof(float);
    float *array = (float *)malloc(file_size);
    if (array == NULL) {
        perror("Memory allocation failed");
        fclose(fp);
        return NULL;
    }

    fread(array, sizeof(float), *size, fp);
    fclose(fp);
    return array;
}


uint8_t *load_8bit_uint_array_from_file(const char *filename, size_t *size)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        perror("Error opening file");
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    *size = file_size / sizeof(uint8_t);
    uint8_t *array = (uint8_t *)malloc(file_size);
    if (array == NULL) {
        perror("Memory allocation failed");
        fclose(fp);
        return NULL;
    }

    fread(array, sizeof(uint8_t), *size, fp);
    fclose(fp);
    return array;
}


int mnist_datasets(MNISTData *dataset, const char *image_path, const char *label_path, int num_images) 
{
    char image_file[256], label_file[256];
    size_t image_size, label_size;

    for (int i = 0; i < num_images; i++) {
        snprintf(image_file, sizeof(image_file), "%s/image%d.bin", image_path, i);
        snprintf(label_file, sizeof(label_file), "%s/label%d.bin", label_path, i);

        dataset[i].image_data = load_32bit_float_array_from_file(image_file, &image_size);
        if (dataset[i].image_data == NULL) {
            fprintf(stderr, "Failed to load image data for index %d\n", i);
            return 1;
        }

        dataset[i].label_data = load_8bit_uint_array_from_file(label_file, &label_size);
        if (dataset[i].label_data == NULL) {
            fprintf(stderr, "Failed to load label data for index %d\n", i);
            return 1;
        }
    }

    return 0;
}
