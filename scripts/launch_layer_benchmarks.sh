#!/bin/bash
sbatch <<EOT
#!/bin/sh
# Cluster config
#SBATCH --job-name=layer_benchmarks
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=32Gb
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=priority
#SBATCH --mail-type=ALL
#SBATCH --mail-user=webstorms@gmail.com
module purge
module load CUDA/11.2.2-GCC-10.3.0
module load Miniconda3/4.9.2
module load cuDNN/8.1.1.33-CUDA-11.2.2

source activate /data/dpag-auditory-neuroscience/kebl6283/fastsnn
echo $CUDA_VISIBLE_DEVICES
nvidia-smi

# Run job
python -u /data/dpag-auditory-neuroscience/kebl6283/PycharmProjects/FastSNN/scripts/run_benchmarks.py
EOT