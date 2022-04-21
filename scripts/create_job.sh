#!/bin/bash
sbatch <<EOT
#!/bin/sh

# Cluster config
#SBATCH --job-name=$1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=16G
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=webstorms@gmail.com

module purge
module load CUDA/11.2.2-GCC-10.3.0
module load Miniconda3/4.7.10
module load cuDNN/8.1.1.33-CUDA-11.2.2

conda activate fastsnn

echo $CUDA_VISIBLE_DEVICES
nvidia-smi

# Run job
python /data/dpag-auditory-neuroscience/kebl6283/PycharmProjects/FastSNN/scripts/train.py ${@:2}
EOT