def compute_batch_statistics(X, momentum=0.9, epsilon=1e-5):
    """
    입력 데이터 X를 사용하여 scale, B, mean, var를 계산합니다.
    Args:
    - X: 입력 데이터 (shape: [N, C, H, W])
    - momentum: 이동 평균을 계산할 때 이전 값에 대한 가중치
    - epsilon: 분모가 0이 되는 것을 방지하기 위한 아주 작은 상수

    Returns:
    - scale: 스케일 파라미터 (shape: [C])
    - B: 시프트 파라미터 (shape: [C])
    - mean: 평균 (shape: [C])
    - var: 분산 (shape: [C])
    """
    N, C, H, W = X.shape
    # 각 채널에 대한 평균과 분산 계산
    mean = np.mean(X, axis=(0, 2, 3), keepdims=False)
    var = np.var(X, axis=(0, 2, 3), keepdims=False)
    # scale은 분산의 제곱근으로, B는 0으로 초기화합니다.
    scale = np.sqrt(var + epsilon)
    B = np.zeros(C)
    # 이전 이동 평균 값을 가져옵니다.
    prev_mean = np.zeros(C)
    prev_var = np.zeros(C)
    # 이전 이동 평균 값이 있는 경우에만 이동 평균 적용
    if momentum > 0:
        scale = momentum * prev_mean + (1 - momentum) * mean
        B = momentum * prev_var + (1 - momentum) * var
    return scale, B, mean, var

# 예제 데이터
B, C, H, W = 1, 3, 224, 224
X = np.random.randn(B, C, H, W)

# Batch Normalization을 위한 변수 계산
scale, B, mean, var = compute_batch_statistics(X)

print("scale:", scale)
print("B:", B)
print("mean:", mean)
print("var:", var)
