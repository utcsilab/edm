#!/bin/bash
mkdir -p prior-samples
exp=00069-train-denoised2-cond-ddpmpp-edm-gpus4-batch180-fp32-csinas2.aspect-foundation-fsese-curated_v0.train-denoised2.cond
net=network-snapshot-004752.pkl
python3 sampling-improved.py m$exp c0 c1 c2 c3 n$net --iterations 300 --batch_size 24 --batch_num 1 --device cuda:1 --resolutions 200 200

