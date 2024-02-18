#include <armnn/ArmNN.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <iostream>
#include <vector>
// 기타 필요한 헤더 파일들...

// MNIST 데이터 로드 관련 함수 및 구조체 정의
typedef struct {
    float *image_data;
    unsigned char *label_data;
} MNISTData;

float *load_fp32_file(const char *filename, size_t *size);
uint8_t *load_uint8_file(const char *filename, size_t *size);
int mnist_datasets(MNISTData *dataset, const char *image_path, const char *label_path, int num_images);

int main() {
    // 모델 및 데이터 경로 설정
    const char* model_path = "./model/mnist12.tflite";
    const char* image_path = "./data/images";
    const char* label_path = "./data/labels";
    int num_images = 10000;
    MNISTData mnist_test[num_images];

    // ARM NN 설정 및 모델 로드
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime = armnn::IRuntime::Create(options);
    armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();
    armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(model_path);
    armnn::NetworkId networkId;
    runtime->LoadNetwork(networkId, std::move(network));

    // 데이터 로드
    if (mnist_datasets(mnist_test, image_path, label_path, num_images) != 0) {
        std::cerr << "Failed to load MNIST data" << std::endl;
        return 1;
    }

    // 추론 실행 및 정확도 계산
    int correct_predictions = 0;
    for (int i = 0; i < num_images; ++i) {
        // 입력 및 출력 텐서 설정
        armnn::InputTensors inputTensors{{0, armnn::ConstTensor(runtime->GetInputTensorInfo(networkId, 0), mnist_test[i].image_data)}};

        // 출력 데이터 배열 준비
        std::vector<float> outputData(runtime->GetOutputTensorInfo(networkId, 0).GetNumElements());
        armnn::OutputTensors outputTensors{{0, armnn::Tensor(runtime->GetOutputTensorInfo(networkId, 0), outputData.data())}};

        // 추론 실행
        runtime->EnqueueWorkload(networkId, inputTensors, outputTensors);

        // 최대 점수와 예측된 라벨 찾기
        int predicted_label = std::distance(outputData.begin(), std::max_element(outputData.begin(), outputData.end()));

        // 정확도 계산
        if (predicted_label == mnist_test[i].label_data[0]) {
            correct_predictions++;
        }
    }

    // 결과 출력
    std::cout << "Accuracy: " << correct_predictions << "/" << num_images << std::endl;

    return 0;
}

float *load_fp32_file(const char *filename, size_t *size)
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


uint8_t *load_uint8_file(const char *filename, size_t *size)
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

        dataset[i].image_data = load_uint8_file(image_file, &image_size);
        if (dataset[i].image_data == NULL) {
            fprintf(stderr, "Failed to load image data for index %d\n", i);
            return 1;
        }

        dataset[i].label_data = load_fp32_file(label_file, &label_size);
        if (dataset[i].label_data == NULL) {
            fprintf(stderr, "Failed to load label data for index %d\n", i);
            return 1;
        }
    }

    return 0;
}
