tvmc run \
--inputs imagenet_cat.npz \
--output predictions.npz  \
--print-time \
--repeat 100 \
resnet50-v2-7-tvm_autotuned.tar

tvmc run \
--inputs imagenet_cat.npz \
--output predictions.npz  \
--print-time \
--repeat 100 \
resnet50-v2-7-tvm.tar