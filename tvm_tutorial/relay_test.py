#!/usr/local/Caskroom/miniforge/base/envs/tvm/bin/python3

from tvm import relay
from tvm.ir.module import IRModule
import numpy as np
import tvm
from tvm.contrib import graph_runtime

# 입력 데이터의 형태와 데이터 타입을 정의합니다.
input_shape = (1, 64)  # 예: 배치 크기가 1이고, 입력 특징이 64개인 경우
input_dtype = "float32"  # 입력 데이터 타입

# Relay에서 사용할 입력 변수를 생성합니다.
data = relay.var("data", relay.TensorType(input_shape, input_dtype))

# 첫 번째 dense layer의 가중치를 생성합니다.
fc1_weight_shape = (32, 64)
fc1_weight = relay.var("fc1_weight", relay.TensorType(fc1_weight_shape, input_dtype))

# 첫 번째 dense layer를 정의합니다.
fc1 = relay.nn.dense(data, fc1_weight)
act1 = relay.nn.relu(fc1)

# 두 번째 dense layer의 가중치를 생성합니다.
fc2_weight_shape = (10, 32)
fc2_weight = relay.var("fc2_weight", relay.TensorType(fc2_weight_shape, input_dtype))

# 두 번째 dense layer를 정의합니다.
fc2 = relay.nn.dense(act1, fc2_weight)
act2 = relay.nn.relu(fc2)

# 간단한 FCNN 모델을 정의합니다.
func = relay.Function(relay.analysis.free_vars(act2), act2)

# Relay 모듈에 함수를 추가합니다.
mod = IRModule()
mod["main"] = func

# 컴파일을 위한 target과 context를 설정합니다.
target = "llvm"  # 예: CPU를 위한 LLVM
ctx = tvm.cpu(0)

# 모델을 컴파일합니다.
with relay.build_config(opt_level=2):
    graph, lib, params = relay.build_module.build(mod, target, params={})
    # opt_level은 최적화 수준을 의미합니다.

# 런타임을 생성하고, 모델을 런타임에 로드합니다.
m = graph_runtime.create(graph, lib, ctx)
m.set_input('data', tvm.nd.array(np.random.uniform(size=input_shape).astype(input_dtype)))
m.set_input(**params)

# 모델을 실행합니다.
m.run()

# 출력을 얻습니다.
tvm_output = m.get_output(0)
