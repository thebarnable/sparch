#!/bin/bash

### example cluster options (refer to https://doc.itc.rwth-aachen.de/display/CC/Using+the+SLURM+Batch+System)
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=1
#SBATCH --job-name=ablation
#SBATCH --output=ablation.%J.txt
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1

### custom setup for python experiments
## setup conda
. $HOME/miniconda3/etc/profile.d/conda.sh
export PATH=$HOME/miniconda3/bin:$PATH
eval "$(conda shell.bash hook)"
conda activate sparch

cd $HOME/1_Projects/sparch

gpu=1

echo "TASK: $SLURM_ARRAY_TASK_ID"
if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then # test run
  python main.py --model RLIF --dataset shd --gpu $gpu --dataset-folder SHD --n-epochs 2 --log true --trials 1 --n-layers 2
elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then # reference (3 layer, 128 neurons, RLIF)
  python main.py --model RLIF --dataset shd --gpu $gpu --dataset-folder SHD --n-epochs 30 --log true --trials 3 --n-layers 2
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]; then # reference - dropout
  python main.py --model RLIF --dataset shd --gpu $gpu --dataset-folder SHD --n-epochs 30 --log true --trials 3 --n-layers 2  --dropout 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]; then # reference - batchnorm
  python main.py --model RLIF --dataset shd --gpu $gpu --dataset-folder SHD --n-epochs 30 --log true --trials 3 --n-layers 2  --normalization none
elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]; then # reference - dropout - batchnorm
  python main.py --model RLIF --dataset shd --gpu $gpu --dataset-folder SHD --n-epochs 30 --log true --trials 3 --n-layers 2 --dropout 0 --normalization none  
elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]; then # reference @ 2 layers
  python main.py --model RLIF --dataset shd --gpu $gpu --dataset-folder SHD --n-epochs 30 --log true --trials 3 --n-layers 1
elif [[ $SLURM_ARRAY_TASK_ID -eq 6 ]]; then # reference @ 2 layers - dropout - batchnorm
  python main.py --model RLIF --dataset shd --gpu $gpu --dataset-folder SHD --n-epochs 30 --log true --trials 3 --n-layers 1 --dropout 0 --normalization none
elif [[ $SLURM_ARRAY_TASK_ID -eq 7 ]]; then # reference + one-spike-only
  python main.py --model RLIF --dataset shd --gpu $gpu --dataset-folder SHD --n-epochs 50 --log true --trials 3 --n-layers 2 --balance true
elif [[ $SLURM_ARRAY_TASK_ID -eq 8 ]]; then # reference @ 2 layers - dropout - batchnorm + one-spike-only
  python main.py --model RLIF --dataset shd --gpu $gpu --dataset-folder SHD --n-epochs 50 --log true --trials 3 --n-layers 1 --dropout 0 --normalization none --balance true
elif [[ $SLURM_ARRAY_TASK_ID -eq 9 ]]; then # reference + one-spike-only + substep 5
  python main.py --model RLIF --dataset shd --gpu $gpu --dataset-folder SHD --n-epochs 50 --log true --trials 3 --n-layers 2 --balance true --substeps 5
elif [[ $SLURM_ARRAY_TASK_ID -eq 10 ]]; then # reference @ 2 layers - dropout - batchnorm + one-spike-only + substep 5
  python main.py --model RLIF --dataset shd --gpu $gpu --dataset-folder SHD --n-epochs 50 --log true --trials 3 --n-layers 1 --dropout 0 --normalization none --balance true --substeps 5
elif [[ $SLURM_ARRAY_TASK_ID -eq 11 ]]; then # reference + one-spike-only + substep 10
  python main.py --model RLIF --dataset shd --gpu $gpu --dataset-folder SHD --n-epochs 50 --log true --trials 3 --n-layers 2 --balance true --substeps 10
elif [[ $SLURM_ARRAY_TASK_ID -eq 12 ]]; then # reference @ 2 layers - dropout - batchnorm + one-spike-only + substep 10
  python main.py --model RLIF --dataset shd --gpu $gpu --dataset-folder SHD --n-epochs 50 --log true --trials 3 --n-layers 1 --dropout 0 --normalization none --balance true --substeps 10
elif [[ $SLURM_ARRAY_TASK_ID -eq 13 ]]; then # reference + one-spike-only + substep 20
  python main.py --model RLIF --dataset shd --gpu $gpu --dataset-folder SHD --n-epochs 50 --log true --trials 3 --n-layers 2 --balance true --substeps 20
elif [[ $SLURM_ARRAY_TASK_ID -eq 14 ]]; then # reference @ 2 layers - dropout - batchnorm + one-spike-only + substep 20
  python main.py --model RLIF --dataset shd --gpu $gpu --dataset-folder SHD --n-epochs 50 --log true --trials 3 --n-layers 1 --dropout 0 --normalization none --balance true --substeps 20
elif [[ $SLURM_ARRAY_TASK_ID -eq 15 ]]; then # reference + one-spike-only + substep 50
  python main.py --model RLIF --dataset shd --gpu $gpu --dataset-folder SHD --n-epochs 50 --log true --trials 3 --n-layers 2 --balance true --substeps 50
elif [[ $SLURM_ARRAY_TASK_ID -eq 16 ]]; then # reference @ 2 layers - dropout - batchnorm + one-spike-only + substep 50
  python main.py --model RLIF --dataset shd --gpu $gpu --dataset-folder SHD --n-epochs 50 --log true --trials 3 --n-layers 1 --dropout 0 --normalization none --balance true --substeps 50
fi

