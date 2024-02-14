import os
import urllib.request
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

def download_onnx_model(url, output_path):
    # model 폴더가 없으면 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # URL에서 파일 다운로드
        with urllib.request.urlopen(url) as response, open(output_path, 'wb') as out_file:
            data = response.read()  # 바이트 데이터 읽기
            out_file.write(data)  # 파일에 저장
        print(f"모델 다운로드 성공: {output_path}")
    except Exception as e:
        print(f"오류 발생: {e}")

def convert_onnx_to_tflite(onnx_file_path, tflite_file_path):
    onnx_model = onnx.load(onnx_file_path)
    tf_rep = prepare(onnx_model)

    # TensorFlow SavedModel 형식으로 모델 내보내기
    saved_model_path = tflite_file_path.replace('.tflite', '_saved_model')
    tf_rep.export_graph(saved_model_path)

    # TensorFlow SavedModel을 TFLite 모델로 변환
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    tflite_model = converter.convert()

    # 변환된 모델을 파일로 저장
    with open(tflite_file_path, 'wb') as f:
        f.write(tflite_model)

# mnist-12 모델 다운로드
url_mnist_12 = "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx"
path_mnist_12_onnx = "./model/mnist12.onnx"
path_mnist_12_tflite = "./model/mnist12.tflite"

download_onnx_model(url_mnist_12, path_mnist_12_onnx)
convert_onnx_to_tflite(path_mnist_12_onnx, path_mnist_12_tflite)
