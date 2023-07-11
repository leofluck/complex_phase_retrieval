#!/bin/bash
#SBATCH --chdir /home/lflueck
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4096
#SBATCH --time 18:00:00
#SBATCH --qos serial
#SBATCH -o results/output_results.out
#SBATCH -e results/error_results.out
#SBATCH --array=0-49

module load gcc
module load python
source SPOC/bin/activate

cd results
python3 cpr_results_2_8.py ${SLURM_ARRAY_TASK_ID}