import numpy as np

def batchnormalization(epsilon, momentum, spatial, X, scale, B, mean, var):
    N, C, H, W = X.shape

    # 정규화
    X_normalized = (X - mean.reshape(1, C, 1, 1)) / np.sqrt(var.reshape(1, C, 1, 1) + epsilon)

    # 스케일링 및 시프트
    out = scale.reshape(1, C, 1, 1) * X_normalized + B.reshape(1, C, 1, 1)

    return out

# 예제 데이터
B, C, H, W = 1, 3, 224, 224
X = np.random.randn(B, C, H, W)
scale = np.ones(C)
B = np.zeros(C)
mean = np.random.randn(C)
var = np.random.rand(C)

# Batch Normalization 실행
bn_out = batchnormalization(1e-5, 0.9, True, X, scale, B, mean, var)


print(bn_out)