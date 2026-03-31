#!/bin/bash
# Training an EDM model initialized with another model

dataset=aspect-motion-free-200x200-bysubject_v0
transfer=00011--uncond-ddpmpp-edm-gpus4-batch64-fp32-fastmri_brain_white_standardsize_192x192_v0
network=network-snapshot-010000.pkl

torchrun --standalone --nproc_per_node=4 train.py --outdir=/csiNAS2/slow/yarefeen/edm-outputs-test --data=/csiNAS2/slow/yarefeen/datasets_for_edm/$dataset/train_v0/all_samples_curated/ --cond=0 --arch=ddpmpp --duration=10 --batch=64 --cbase=128 --cres=1,1,2,2,2,2,2 --lr=1e-4 --ema=0.1 --dropout=0.0 --desc=$dataset-transfer-from-$transfer --tick=1 --dump=100  --precond edm --transfer=/csiNAS2/slow/yarefeen/edm-outputs-test/$transfer/$network

#torchrun --standalone --nproc_per_node=4 train.py --outdir=/csiNAS2/slow/yarefeen/edm-outputs-test --data=/csiNAS2/slow/yarefeen/datasets_for_edm/$dataset/train_v0/all_samples_curated/ --cond=0 --arch=ddpmpp --duration=10 --batch=64 --cbase=128 --cres=1,1,2,2,2,2,2 --lr=1e-4 --ema=0.1 --dropout=0.0 --desc=$dataset --tick=1 --dump=100 --seed=2023 --precond edm --transfer
