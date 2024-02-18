#!/bin/bash

kernel_ops=20
model=$1
server_ip=$2
server_port=$3

repeat=8
evals=1500

mkdir tune
mkdir log
    
for ((conv=-1; conv <= $kernel_ops; conv++)); do
    ./chacha -m optimize -o ${model}.tflite -r ${repeat} -e ${evals} -t ./tune/${model}/${conv}.json -c ${conv} -l ./log/${mod}/log_${conv}.json -s ${server_ip} -p ${server_port}
done

python3 integrate_convolution.py -t ./tune/${model} -o ./${model}-for-eval.json
