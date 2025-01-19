#!/bin/bash -l

#SBATCH --time=4:00:00
#SBATCH --mem=40G
#SBATCH --output=output/d_%a.out
#SBATCH --gres=gpu:1
#SBATCH --array=1-306
#SBATCH --partition=gpu-v100-32g,gpu-a100-80g,gpu-h100-80g-short

#source activate kuramoto_env
source activate crosspy_env

index=$SLURM_ARRAY_TASK_ID
rootdirectory="/scratch/nbe/grr_epilepsy/MEG_spont"

filepath=$(find "${rootdirectory}" -type f \( -name "*.npy" -o -name "*.mat" \) | sort -n | sed -n "${index}p")

python run_CDL_MEG_spont.py --filepath=$filepath

# https://scicomp.aalto.fi/triton/quickstart/jobs/
# in terminal: sbatch batch_CDL_MEG_spont.sh
