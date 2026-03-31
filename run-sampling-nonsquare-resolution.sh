#!/bin/bash

x_resolution=200
y_resolution=200
sampling_iterations=300
batch_size=16
num_batches=1
cuda='cuda:3'


model=network-snapshot-000304.pkl
experiment=00032-train-cond-ddpmpp-edm-gpus3-batch39-fp32-aspect-larger-curated_v2.train.cond
#experiment=00029-train-uncond-ddpmpp-edm-gpus4-batch48-fp32-aspect-larger-curated_v1.train

python3 sampling-yamin.py $num_batches $batch_size $x_resolution $y_resolution $sampling_iterations $cuda $experiment $model -1
