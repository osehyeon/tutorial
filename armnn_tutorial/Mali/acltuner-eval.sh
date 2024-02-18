#!/bin/bash

model=$1

models="${model}.tflite"
tune_dir="tune/${model}"

./chacha -m 1 -o ${models} -c -1 -l ./${tune_dir}/tune.cfg -a ./${tune_dir}/100-best-kernel.csv -e 1000
