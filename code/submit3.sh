#!/bin/bash
#SBATCH --account=def-wperrie
#SBATCH --nodes=2
#SBATCH --gpus-per-node=v100l:4   
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --mem=0               # Request the full memory of the node
#SBATCH --time=48:00:00
module purge
module load python/3.10
module load scipy-stack
module load cuda/11.4 
module load cudnn

##virtualenv --no-download ~/mypython/env
source ~/mypython/env/bin/activate
###virtualenv --no-download tensorflow
source tensorflow/bin/activate
module list

###
#import os
#os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
#os.environ["TF_CPP_VMODULE"]="gpu_process_state=10,gpu_cudamallocasync_allocator=10"
#a = tf.zeros([], tf.float32)
#python -c 'import tensorflow'
###
##pip install --no-index tensorflow==2.8
##pip install --no-index keras
##pip install --no-index tqdm
##pip install --no-index xarray
##pip install  --no-index numpy 
###pip install h5io h5py h5netcdf --no-index
python train2.py
