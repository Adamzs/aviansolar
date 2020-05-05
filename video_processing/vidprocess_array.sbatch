#!/bin/bash
#SBATCH --job-name=avian_%A-%a
#SBATCH --output=vidprocess_%A_%a.out
#SBATCH --error=vidprocess_%A_%a.err
#SBATCH --array=3
#SBATCH --time=00:30:00
#SBATCH --partition=sball
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-user=szymanski@anl.gov

module load anaconda3/5.2.0
source activate opencv4

python main_app_fileoutput_batcharray-avian.py -write_images /lcrc/project/aviansolar/avian-solar-data/videos/dirs_list.csv /lcrc/project/aviansolar/avian-solar-data/videos/ /lcrc/project/aviansolar/avian-solar-data/image_data/ $SLURM_ARRAY_TASK_ID
