import numpy as np
import os,sys,time,datetime
from os.path import expanduser
import argparse


import subprocess
# git_rev = subprocess.Popen("git rev-parse --short HEAD", shell=True, stdout=subprocess.PIPE, universal_newlines=True).communicate()[0]
# git_branch = subprocess.Popen("git symbolic-ref --short -q HEAD", shell=True, stdout=subprocess.PIPE, universal_newlines=True).communicate()[0]


class Globals:

    git_rev = ''
    git_branch = None
    timestamp = None
    parser = None
    args_global = None
    NUM_PAR_SAMPLER = None
    SAMPLES_PER_PROC = None
    EVAL_VAL_EVERY_EP = None
    f_mean = None
    DTYPE = None
    loss_type = None
    loss_action = None

    @staticmethod
    def update_globals(gs_args):
        Globals.git_rev = ''
        Globals.git_branch = ''

        Globals.timestamp = time.time()
        Globals.timestamp = datetime.datetime.fromtimestamp(int(Globals.timestamp)).strftime('%Y-%m-%d %H-%M-%S')




        # Set random seed
        #seed = 123
        #np.random.seed(seed)
        #tf.set_random_seed(seed)

        Globals.parser = argparse.ArgumentParser(description="argument for GraphSAINT training")
        Globals.parser.add_argument("--num_cpu_core",default=20,type=int,help="Number of CPU cores for parallel sampling")
        Globals.parser.add_argument("--log_device_placement",default=False,action="store_true",help="Whether to log device placement")
        Globals.parser.add_argument("--data_prefix",required=False,type=str,help="prefix identifying training data")
        Globals.parser.add_argument("--dir_log",default=".",type=str,help="base directory for logging and saving embeddings")
        Globals.parser.add_argument("--gpu",default="-1234",type=str,help="which GPU to use")
        Globals.parser.add_argument("--eval_train_every",default=1,type=int,help="How often to evaluate training subgraph accuracy")
        Globals.parser.add_argument("--train_config",required=False,type=str,help="path to the configuration of training (*.yml)")
        Globals.parser.add_argument("--dtype",default="s",type=str,help="d for double, s for single precision floating point")
        Globals.parser.add_argument("--timeline",default=False,action="store_true",help="to save timeline.json or not")
        Globals.parser.add_argument("--tensorboard",default=False,action="store_true",help="to save data to tensorboard or not")
        Globals.parser.add_argument("--dualGPU",default=False,action="store_true",help="whether to distribute the model to two GPUs")
        Globals.parser.add_argument("--cpu_eval",default=False,action="store_true",help="whether to use CPU to do evaluation")
        Globals.parser.add_argument("--saved_model_path",default="",type=str,help="path to pretrained model file")
        Globals.parser.add_argument("--loss_type", default="node", type=str, help="which loss to use (node / edge)")
        Globals.parser.add_argument("--loss_action", default="mul", type=str, help="which loss action to use (mul / cat)")
        Globals.args_global = Globals.parser.parse_args(gs_args)


        Globals.NUM_PAR_SAMPLER = Globals.args_global.num_cpu_core
        Globals.SAMPLES_PER_PROC = -(-200 // Globals.NUM_PAR_SAMPLER) # round up division

        Globals.EVAL_VAL_EVERY_EP = 1       # get accuracy on the validation set every this # epochs


        # auto choosing available NVIDIA GPU
        gpu_selected = Globals.args_global.gpu
        if False and gpu_selected == '-1234':
            # auto detect gpu by filtering on the nvidia-smi command output
            gpu_stat = subprocess.Popen("nvidia-smi",shell=True,stdout=subprocess.PIPE,universal_newlines=True).communicate()[0]
            gpu_avail = set([str(i) for i in range(8)])
            for line in gpu_stat.split('\n'):
                if 'python' in line:
                    if line.split()[1] in gpu_avail:
                        gpu_avail.remove(line.split()[1])
                    if len(gpu_avail) == 0:
                        gpu_selected = -2
                    else:
                        gpu_selected = sorted(list(gpu_avail))[0]
            if gpu_selected == -1:
                gpu_selected = '0'
            Globals.args_global.gpu = int(gpu_selected)
        if str(gpu_selected).startswith('nvlink'):
            os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_selected).split('nvlink')[1]
        elif int(gpu_selected) >= 0:
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_selected)
            GPU_MEM_FRACTION = 0.8
        else:
            os.environ["CUDA_VISIBLE_DEVICES"]="-1"
        Globals.args_global.gpu = int(Globals.args_global.gpu)

        # global vars

        Globals.f_mean = lambda l: sum(l)/len(l)

        Globals.DTYPE = "float32" if Globals.args_global.dtype=='s' else "float64"      # NOTE: currently not supporting float64 yet
