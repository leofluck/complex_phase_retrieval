#!/bin/bash
#SBATCH --chdir /home/lflueck
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4096
#SBATCH --time 48:00:00
#SBATCH --qos serial
#SBATCH -o output_SGD_par.out
#SBATCH -e error_SGD_par.out
#SBATCH --array=0-499

module load gcc
module load python
source SPOC/bin/activate

python3 cpr_pickle.py ${SLURM_ARRAY_TASK_ID}