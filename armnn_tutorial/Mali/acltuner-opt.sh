#!/bin/bash

repeat=1
model=$1
evals=8

mkdir tune
models_dir="./"
models="${model}.tflite"
tune_dir="tune/${model}"

cd tune
mkdir ${model}
cd ..

echo ./chacha -m 0 -o ${models_dir}/${models} -c -1 -l ./${tune_dir}/${models}-opt-default.csv 
./chacha -m 0 -o ${models_dir}/${models} -c -1 -l ./${tune_dir}/${models}-opt-default.csv -e ${evals}

for ((mk=0; mk < 3; mk++)); do
    for ((alg=0; alg < 3; alg++)); do
        echo ./chacha -m 2 -o ${models_dir}/${models} -c ${alg}${mk} -l ./${tune_dir}/${models}-opt-${alg}${mk}.csv
        ./chacha -m 2 -o ${models_dir}/${models} -c ${alg}${mk} -l ./${tune_dir}/${models}-opt-${alg}${mk}.csv -e ${evals}
    done
done

python3 integrate_convolution.py --log ./${tune_dir} --tune ./${tune_dir}/tune.cfg


# echo ./chacha -m 0 -o ../execute_for_tflite_with_xnnpack/${n}.tflite -c
# echo ./chacha -m 1 -o ${models_dir}/${models} -c -1 -l ./${tune_dir}/tune.cfg --log_log ./${tune_dir}/best-kernel.csv
