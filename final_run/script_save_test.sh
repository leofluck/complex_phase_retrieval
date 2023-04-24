#!/bin/bash
#SBATCH --chdir /home/lflueck
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4096
#SBATCH --time 1:00:00
#SBATCH --qos serial
#SBATCH -o output_test_save.out
#SBATCH -e error_test_save.out

module load gcc
module load python
source SPOC/bin/activate

python3 test_save.py