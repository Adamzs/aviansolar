#!/bin/bash
#SBATCH --job-name=vidprocess_run1_sbatch
#SBATCH --output=vidprocess_run1_sbatch.out
#SBATCH --error=vidprocess_run1_sbatch.err
#SBATCH --time=08:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000

module load Anaconda3/5.0.1
source activate opencv
module load ffmpeg
python main_app_fileoutput.py 2017-09-29
