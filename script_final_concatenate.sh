#!/bin/bash
#SBATCH --chdir /home/lflueck
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4096
#SBATCH --time 00:30:00
#SBATCH --qos serial
#SBATCH -o output_final_concatenate.out
#SBATCH -e error_final_concatenate.out

module load gcc
module load python
source SPOC/bin/activate

python3 cpr_final.py ${SLURM_ARRAY_TASK_ID}