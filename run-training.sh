#!/bin/bash
#
dataset_name=fastmri_brain_white_standardsize_v0_num_coils12_res0.9_dim2/train
dataset_path=/csiNAS3/yarefeen/accelerated_recon_hypothesis/src/datasets/for_edm
outdir_path=/csiNAS3/yarefeen/accelerated_recon_hypothesis/src/edm_models

torchrun --standalone --nproc_per_node=2 train.py --outdir=$outdir_path/$dataset_name --data=$dataset_path/$dataset_name --cond=0 --arch=ddpmpp --duration=10 --batch=64 --cbase=128 --cres=1,1,2,2,2,2,2 --lr=1e-4 --ema=0.1 --dropout=0.0 --desc=$dataset_name --snap 25 --tick=1 --dump=100 --seed=2023 --precond edm
