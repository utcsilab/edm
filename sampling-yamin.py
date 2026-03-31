#!/usr/bin/env python
# coding: utf-8

# generating prior samples for a particular model.  using as a way to check progress of the model

# INPUTS: num_samples, num_batch, M , N, num_steps, device, <net path>

import sys
import os 
import torch
from torch_utils import distributed as dist
import dnnlib
import pickle
import numpy as np

from sampling_funcs import StackedRandomGenerator,ODE_motion_sampler

num_samples = int(sys.argv[1]) # number of samples that we want to generate
num_batch   = int(sys.argv[2]) # bath size
M           = int(sys.argv[3]) 
N           = int(sys.argv[4]) # image dimensions

#- sampling parameters
num_steps = int(sys.argv[5]) 
sigma_max = 5.0
sigma_min = 0.002

rho = 7 # for time-step discretization

device = sys.argv[6]

base_path = '/csiNAS2/slow/yarefeen/edm-outputs-test'
dataset   = sys.argv[7]
model     = sys.argv[8]

net_save = base_path + '/' + dataset + '/' + model

if dist.get_rank() != 0:
        torch.distributed.barrier()
        
dist.print0(f'Loading network from "{net_save}"...')
with dnnlib.util.open_url(net_save, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)

print(net.label_dim)
#- setting class labels
if len(sys.argv) <= 9:
    class_labels = None
elif int(sys.argv[9])>=0:
    # use a specific class
    class_labels = torch.eye(net.label_dim, device=device)[int(sys.argv[9])*torch.ones(num_batch,dtype=torch.long,device=device)]
else:
    # use random classes
    class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[num_batch], device=device)]
print(type(num_samples))
print(num_samples, num_batch, M ,N)
print(class_labels)

#-setting time-step discretization
step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

all_x = np.zeros((num_batch,M,N,num_steps,num_samples),dtype = complex)

# - should probably do this in batches later
for num_sample in range(num_samples):
    print('SAMPLE: %d / %d' % (num_sample+1, num_samples))
    rnd = StackedRandomGenerator(device, [1]) # use the current sample number as the seeda
    
    #-generating the gaussian sample which will be 'denoised' to our desired sample
#     latents = rnd.randn([1, 2, M, N], device=device)
    latents = torch.randn((num_batch,2,M,N),device=device)
    x = latents.to(torch.float64) * t_steps[0] 
    # scale initial noise image to have same variance as starting sigma
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1 # for tqdm
        print('   step: %d/%d' % (i+1,num_steps))
        #-getting scoare with denoising
        denoised = net(x, t_cur, class_labels).to(torch.float64)

        score = (x - denoised) / t_cur
        
        #-performing the update
        x = x + (t_next - t_cur) * score
        all_x[:,:,:,i,num_sample] = torch.view_as_complex(x.permute(0,-2,-1,1).contiguous()).squeeze().cpu()

all_x_rsh = np.transpose(all_x,axes=(0,4,1,2,3))
all_x_rsh = np.reshape(all_x_rsh,(num_samples*num_batch,M,N,num_steps))

import cfl
#cfl.writecfl('tmp/testprior_allsteps_%dx%d_iters%d_%s_%s' % (M,N,num_steps,dataset,model), all_x_rsh)
cfl.writecfl('tmp/testprior_justsampl_%dx%d_iters%d_%s_%s' % (M,N,num_steps,dataset,model),all_x_rsh[:,:,:,-1])
