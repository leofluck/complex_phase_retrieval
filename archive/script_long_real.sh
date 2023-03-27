#!/bin/bash
#SBATCH --chdir /home/lflueck
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4096
#SBATCH --time 10:00:00
#SBATCH --partition serial
#SBATCH -o output_SGD_long_real.out
#SBATCH -e error_SGD_long_real.out

module load gcc
module load python
source SPOC/bin/activate

python3 cpr_opti.py