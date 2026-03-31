import sys, os, torch, dnnlib, pickle, numpy as np, argparse, tqdm, cfl
from torch_utils import distributed as dist
from sampling_funcs import StackedRandomGenerator

def parse():
    parser = argparse.ArgumentParser(description="Generating prior samples for a range of x/y res, iterations, and experiments" )
    parser.add_argument('experiments_models_labels',type=str,nargs='+',help='experiments, associated labels we want to sample for that experiment, and associated models [m<exp1> c<label1> c<label2>... n<net1> n<net2>... m<exp2> c<label1> c<label2>... n<net1> n<net2>... m<exp3>...]. label=-2 -> unconditional sampling, label=-2 -> no class conditioning, label=-1 -> random conditional sampling, label>=0 -> conditonal sampling with specified class')
    parser.add_argument('--batch_size',type=int,default=16,help='batch size for prior sample generation')
    parser.add_argument('--batch_num',type=int,default=1,help='number of batches')
    parser.add_argument('--resolutions',type=int,nargs='+',default=[200,200],help='resolution in (x,y) must be an even number of values')
    parser.add_argument('--iterations', type=int, nargs='+', default=[300], help='sampling iterations')
    parser.add_argument('--device', type=str, default='cuda:3', help='gpu number')
    parser.add_argument('--base_path', type=str, default='/csiNAS2/slow/yarefeen/edm-outputs-test', help='base path to where models are saved')
    return parser.parse_args()

def parse_resolutions(resolutions):
    '''
    - Helper function that splits a resolutions array into xres and yres. 
    - First check if resolutions array is even
    - Assume [x1 y1 x2 y2 ...] 
    '''
    if len(resolutions) % 2 != 0:
        sys.exit('please enter an even number of resolutions for xres and yres')
    
    output_resolutions = []
    cur_group = []
    for ii, res in enumerate(resolutions):
        ii += 1
        cur_group.append(res)

        if ii % 2 == 0:
            output_resolutions.append(cur_group)
            cur_group = []
    return output_resolutions

def parse_experiments_models_labels(x):
    '''
    - Helper function that helps parse experiments_models input arrays
    '''
    #- first parsing experiment and model
    output_experiments_models_classes = []
    for ss, string in enumerate(x):
        # - checking whether the string we are currently reading in is an experiment or model or class label
        if string[0]=='m': 
            if ss: #if we've at least completed 1 pass, i.e. ss > 0
                for clas in cur_classes:
                    for model in cur_models:
                        output_experiments_models_classes.append([cur_experiment,clas,model])
            cur_experiment = string[1:]
            cur_classes= []
            cur_models = []
        elif string[0]=='n':
            cur_models.append(string[1:])
        elif string[0]=='c':
            cur_classes.append(int(string[1:]))

    # - need to also do for the last model
    for clas in cur_classes:
        for model in cur_models:
            output_experiments_models_classes.append([cur_experiment,clas,model])


    return output_experiments_models_classes

def generate_parameters(args):
    '''
    Generating parameters for prior sampling.  Each parameter set should have
    [
        experiment
        model
        resolution
        iteration
    ]
    '''
     
    #- now creating parameter set
    parameters = []
    for experiment_model_label in parse_experiments_models_labels(args.experiments_models_labels):
        for iteration in args.iterations:
            for resolution in parse_resolutions(args.resolutions):
                parameters.append(
                        {'experiment_model_label':experiment_model_label,
                         'iteration':iteration,
                         'resolution':resolution
                        }
                        )
    return parameters 

def print_dict(dictionary):
    '''
    print keys and values in a dictionary
    '''
    for key, value in dictionary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    print('...STARTING...')
    args = parse()
    parameters = generate_parameters(args) 
    
    base_path = args.base_path
    batch_size = args.batch_size
    batch_num = args.batch_num
    device = args.device

    print('~~~Common Parameters~~~')
    print('base_path %s || batch_size %d || batch_num %d || device %s'%(base_path,batch_size,batch_num,device))

    # - hardcoded sampling parameters
    sigma_max = 5.0
    sigma_min = 0.002
    rho = 7 

    # - ablation loop
    all_prior_samples = []
    for pp, par in enumerate(parameters):
        print('~~~Parameter %d/%d~~~'%(pp+1,len(parameters)))
        print_dict(par)
        experiment = par["experiment_model_label"][0]
        label      = par["experiment_model_label"][1]
        model      = par["experiment_model_label"][2]
        iteration  = par["iteration"]
        xres       = par["resolution"][0]
        yres       = par["resolution"][1]

        #-loading the network
        model_path = os.path.join(base_path,experiment,model)

        if dist.get_rank() != 0:
            torch.distributed.barrier()
        
        dist.print0(f'  -> loading network from "{model_path}"...')
        with dnnlib.util.open_url(model_path, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f)['ema'].to(device)

        #- generating the class labels
        if label == -2:
            # unconditional sampling
            class_labels = None
        elif label == -1:
            # random class for each sample in the batch
            class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]
        elif label >=0:
            # using a specific class
            class_labels = torch.eye(net.label_dim, device=device)[label*torch.ones(batch_size,dtype=torch.long,device=device)]
        else:
            # incorrect label
            sys.exit('label of %d is invalid')

        #-setting time-step discretization
        step_indices = torch.arange(iteration, dtype=torch.float64, device=device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (iteration- 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        #-performing prior sampling
        prior_samples = np.zeros((batch_size,xres,yres,batch_num),dtype = complex)
        for batch in range(batch_num):
            print('  -> batch: %d / %d' % (batch+1, batch_num))
            rnd = StackedRandomGenerator(device, [batch]) 
            latents = torch.randn((batch_size,2,xres,yres),device=device)
            x = latents.to(torch.float64) * t_steps[0] 
    
            for i, (t_cur, t_next) in enumerate(zip(tqdm.tqdm(t_steps[:-1]), t_steps[1:])): # 0, ..., N-1 # for tqdm
                denoised = net(x, t_cur, class_labels).to(torch.float64)
                score = (x - denoised) / t_cur 
                x = x + (t_next - t_cur) * score

            prior_samples[:,:,:,batch] = torch.view_as_complex(x.permute(0,-2,-1,1).contiguous()).squeeze().cpu()
        all_prior_samples.append(prior_samples)

# saving
ctr = 0
directory_exists = 0

while not directory_exists:
    save_path = os.path.join('prior-samples','experiment%d'%ctr)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        directory_exists=1
    ctr+=1

for pp, prior_samples in enumerate(all_prior_samples):
    parampath = os.path.join(save_path,'parameter%d'%pp)
    if not os.path.isdir(parampath): os.mkdir(parampath)
    cfl.writecfl(os.path.join(parampath,'samples'),prior_samples)

np.save(os.path.join(save_path,'parameters.npy',),parameters)
os.system('cp %s %s'%(sys.argv[0],save_path))

commandline_string = ''
for arg in sys.argv:
    commandline_string = commandline_string + arg  + ' '
os.system('echo %s > %s'%(commandline_string,os.path.join(save_path,'command.txt')))
