%reset -f
import os, json, matplotlib.pyplot as plt, numpy as np, tqdm, torch, dnnlib, pickle
from sigpy.plot import ImagePlot as iplt
from torch_utils import distributed as dist
from sampling_funcs import StackedRandomGenerator,ODE_motion_sampler
# Parameters
plot_loss = False
# Paths
path = '/csiNAS3/yarefeen/accelerated_recon_hypothesis/src/edm_models/fastmri_brain_white_standardsize_v0_num_coils12_res1.0_dim2/train/00000-train-uncond-ddpmpp-edm-gpus4-batch32-fp32-fastmri_brain_white_standardsize_v0_num_coils12_res1.0_dim2/train'
data_list = []
# @@  Loading loss @@
stats_path = os.path.join(path,'stats.jsonl')
with open(stats_path, 'r') as file:
    for line in tqdm.tqdm(file):
        json_obj = json.loads(line.strip())
        data_list.append(json_obj)
losses = []
for data in data_list:
    losses.append(data['Loss/loss']['mean'])
losses = np.array(losses)
if plot_loss:
    print(losses.shape)
    plt.figure()
    plt.plot(losses)
    plt.show()
# @@ Prior Sampling @@
num_samples = 1  
num_batch   = 8  
M = 320
N = 320
class_labels = None
num_steps = 300
sigma_max = 5.0
sigma_min = 0.002
rho = 7 
device = 'cuda:0'
net = 'network-snapshot-005017.pkl'
net_save = os.path.join(path,net)
if dist.get_rank() != 0:
        torch.distributed.barrier()
with dnnlib.util.open_url(net_save, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)
#-setting time-step discretization
step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) 
all_x = np.zeros((num_batch,M,N,num_steps,num_samples),dtype = complex)
# - should probably do this in batches later
for num_sample in range(num_samples):
    print('SAMPLE: %d / %d' % (num_sample+1, num_samples))
    rnd = StackedRandomGenerator(device, [1]) 
    latents = torch.randn((num_batch,2,M,N),device=device)
    x = latents.to(torch.float64) * t_steps[0] 
    for i, (t_cur, t_next) in enumerate(zip(tqdm.tqdm(t_steps[:-1]), t_steps[1:])): 
        denoised = net(x, t_cur, class_labels).to(torch.float64)
        score = (x - denoised) / t_cur
        x = x + (t_next - t_cur) * score
        all_x[:,:,:,i,num_sample] = torch.view_as_complex(x.permute(0,-2,-1,1).contiguous()).squeeze().cpu()

