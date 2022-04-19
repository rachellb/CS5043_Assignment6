#!/bin/bash
#
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
# memory in MB
#SBATCH --mem=1024
#SBATCH --output=results/molecule_GRU_%04a_stdout.txt
#SBATCH --error=results/molecule_GRU_%04a_stderr.txt
#SBATCH --time=24:00:00
#SBATCH --job-name=molecule_GRU_exp
#SBATCH --mail-user=rachel.l.bennett-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504303/aml/CS5043_Assignment6
#SBATCH --array=0-4

. /home/fagg/tf_setup.sh
conda activate tf


python hw6_base.py @parameters.txt --label "GRU" --exp_index $SLURM_ARRAY_TASK_ID


