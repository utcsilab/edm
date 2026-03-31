#!/bin/bash

# sample prior from a model that I'm currently training, will probably use to debug how a model is doing

# num_samples num_batch M N num_steps device dataset model
#python3 sampling-yamin.py 1 25 191 191 300 'cuda:2' 00008-train_v0-uncond-ddpmpp-edm-gpus2-batch50-fp32-aspect-motion-free-200x200_v0_to_192x192_v0_curated_v0_train_v0 network-snapshot-005300.pkl

max_resolution=300
min_resolution=100
sampling_iterations=300
batch_size=16
num_batches=1
cuda='cuda:2'
step=11
model=network-snapshot-010000.pkl

python3 sampling-yamin.py $num_batches $batch_size 192 192 $sampling_iterations $cuda 00008-train_v0-uncond-ddpmpp-edm-gpus2-batch50-fp32-aspect-motion-free-200x200_v0_to_192x192_v0_curated_v0_train_v0 $model

for (( i=$min_resolution; i <= $max_resolution; i+=$step))
do
    python3 sampling-yamin.py $num_batches $batch_size $i $i $sampling_iterations $cuda 00008-train_v0-uncond-ddpmpp-edm-gpus2-batch50-fp32-aspect-motion-free-200x200_v0_to_192x192_v0_curated_v0_train_v0 $model
done

