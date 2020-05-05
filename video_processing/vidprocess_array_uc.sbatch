#!/bin/bash
#SBATCH --job-name=vid_process_job
#SBATCH --output=vidprocess_%A_%a.out
#SBATCH --error=vidprocess_%A_%a.err
#SBATCH --array=1-39
#SBATCH --time=12:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=2000

module load Anaconda3/5.0.1
source activate opencv
module load ffmpeg
python main_app_fileoutput_batcharray.py $SLURM_ARRAY_TASK_ID
