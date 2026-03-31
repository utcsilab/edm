#!/bin/bash
torchrun --standalone --nproc_per_node=4 train.py --outdir=/csiNAS2/slow/yarefeen/edm-outputs-test --data=/csiNAS2/slow/yarefeen/datasets_for_edm/aspect-foundation-curated-fse-se_v0/train-denoised2 --cond=1 --arch=ddpmpp --duration=10 --batch=180 --cbase=128 --cres=1,1,2,2,2,2,2 --lr=1e-4 --ema=0.1 --dropout=0.0 --desc=csinas2.aspect-foundation-fsese-curated_v0.train-denoised2.cond --tick=1 --dump=100 --seed=2023 --precond edm

# Command below producing noisy samples on somewhat larger neonatal dataset
#torchrun --standalone --nproc_per_node=2 train.py --outdir=/csiNAS2/slow/yarefeen/edm-outputs-test --data=/csiNAS/yamin/edm-datasets/aspect-brett-yamin-curated_v0/train-corrected --cond=0 --arch=ddpmpp --duration=10 --batch=48 --cbase=128 --cres=1,1,2,2,2,2,2 --lr=1e-4 --ema=0.1 --dropout=0.0 --desc=brett-yamin-curated  --tick=1 --dump=100 --seed=2023 --precond edm

# Command I ran for train / test split experiment
#torchrun --standalone --nproc_per_node=4 train.py --outdir=/csiNAS2/slow/yarefeen/edm-outputs-test --data=/csiNAS2/slow/yarefeen/datasets_for_edm/aspect-motion-free-200x200_v0_to_192x192_v0_curated_v0/train_v0 --cond=0 --arch=ddpmpp --duration=10 --batch=64 --cbase=128 --cres=1,1,2,2,2,2,2 --lr=1e-4 --ema=0.1 --dropout=0.0 --desc=aspect-motion-free-200x200_v0_to_192x192_v0_curated_v0_train_v0 --tick=1 --dump=100 --seed=2023 --precond edm

# ORIGINAL COMMAND
#torchrun --standalone --nproc_per_node=4 train.py --outdir=/csiNAS2/slow/brett/edm_outputs (CHANGE) --data=/csiNAS2/slow/brett/fastmri_brain_2000_samples (CHANGE) --cond=0 --arch=ddpmpp --duration=10 --batch=40 --cbase=128 --cres=1,1,2,2,2,2,2 --lr=1e-4 --ema=0.1 --dropout=0.0 --desc=2000_samples --tick=1 --dump=100 --seed=2023 --precond edm
# BRETT's ORIGINAL COMMAND THAT HE SHARED WITH ME
