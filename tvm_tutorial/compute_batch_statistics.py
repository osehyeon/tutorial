import numpy as np

def compute_batch_statistics(X, prev_mean, prev_var, momentum=0.9, epsilon=1e-5):
    N, C, H, W = X.shape
    current_mean = np.mean(X, axis=(0, 2, 3), keepdims=False)
    current_var = np.var(X, axis=(0, 2, 3), keepdims=False)

    updated_mean = momentum * prev_mean + (1 - momentum) * current_mean
    updated_var = momentum * prev_var + (1 - momentum) * current_var

    scale = np.sqrt(updated_var + epsilon)
    B = np.zeros(C)

    return scale, B, updated_mean, updated_var

def process_data_and_compute_statistics(dataset, batch_size, momentum=0.9, epsilon=1e-5):
    C, H, W = dataset.shape[1], dataset.shape[2], dataset.shape[3]
    prev_mean = np.zeros(C)
    prev_var = np.zeros(C)

    for i in range(0, len(dataset), batch_size):
        batch_X = dataset[i:i+batch_size]
        scale, B, prev_mean, prev_var = compute_batch_statistics(batch_X, prev_mean, prev_var, momentum, epsilon)
        print(f"Batch {i//batch_size + 1}:")
        print(f"Scale: {scale}")
        print(f"B: {B}")
        print(f"Updated Mean: {prev_mean}")
        print(f"Updated Var: {prev_var}\n")

    return scale, B, prev_mean, prev_var

# 함수 호출
data_size = 100  # 총 샘플 수
batch_size = 10  # 배치 크기
C, H, W = 1, 224, 224  # 채널, 높이, 너비
dataset = np.random.randn(data_size, C, H, W)  # 랜덤 데이터셋 생성

scale, B, mean, var = process_data_and_compute_statistics(dataset, batch_size)
