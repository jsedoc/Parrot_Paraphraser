#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=parrot
#SBATCH --mail-type=END
#SBATCH --mail-user=js11531@nyu.edu
#SBATCH --output=slurm_out/parrot-%j.out

echo "hostname:" `hostname`

singularity exec --nv --overlay ~/labshare/Parrot_Paraphraser/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif  /bin/bash -c "
source /ext3/env.sh
conda activate parrot
python parrot/paraphrase.py $FILE
"
