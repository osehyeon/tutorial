#!/bin/bash

model=$1
server_ip=$2
server_port=$3

repeat=1
evals=1500

./chacha -m execute -o ./${model}.tflite -r ${repeat} -e ${evals} -t ./${model}-for-eval.json -l ./log/${model}-for-eval.log.json -s ${server_ip} -p ${server_port}
