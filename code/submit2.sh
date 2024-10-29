#!/bin/bash
#SBATCH --account=def-wperrie
#SBATCH --gres=gpu:4 
###SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G      # increase as needed
#SBATCH --time=24:00:00

module load python/3.10 scipy-stack
module load cuda cudn
##virtualenv --no-download ~/mypython/env
source ~/mypython/env/bin/activate
###virtualenv --no-download tensorflow
source tensorflow/bin/activate
##pip install --no-index tensorflow==2.8
##pip install --no-index keras
##pip install --no-index tqdm
##pip install --no-index xarray
##pip install  --no-index numpy 
###pip install h5io h5py h5netcdf --no-index
python train.py
