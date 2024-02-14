import os
import numpy as np
import urllib.request

# 데이터셋 다운로드 함수
def download_dataset(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        print(f"Downloading dataset to {save_path}...")
        urllib.request.urlretrieve(url, save_path)
    else:
        print("Dataset already downloaded.")

def save_images_and_labels(images, labels, images_dir, labels_dir):
    """ 이미지와 라벨을 저장하는 함수 """
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for i, (image, label) in enumerate(zip(images, labels)):
        image_path = os.path.join(images_dir, f'image{i}.bin')
        label_path = os.path.join(labels_dir, f'label{i}.bin')
        image.astype(np.float32).tofile(image_path)
        label.astype(np.uint8).tofile(label_path)

# MNIST 데이터셋 URL 및 저장 경로 설정
dataset_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
dataset_path = './data/mnist.npz'

# 데이터셋 다운로드
download_dataset(dataset_url, dataset_path)

# 데이터 로드
with np.load(dataset_path, allow_pickle=True) as data:
    test_images = data['x_test']
    test_labels = data['y_test']

# 이미지 및 라벨 저장 경로 설정 및 저장
images_dir = './data/images'
labels_dir = './data/labels'
save_images_and_labels(test_images, test_labels, images_dir, labels_dir)

print(f"MNIST data saved in {images_dir} and {labels_dir}.")
