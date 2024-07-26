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
  python run_exp.py --model_type RLIF --dataset_name shd --gpu $gpu --data_folder SHD --nb_epochs 2 --log_tofile true --trials 1 --nb_layers 3
elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then # reference (3 layer, 128 neurons, RLIF)
  python run_exp.py --model_type RLIF --dataset_name shd --gpu $gpu --data_folder SHD --nb_epochs 30 --log_tofile true --trials 3 --nb_layers 3
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]; then # reference - dropout
  python run_exp.py --model_type RLIF --dataset_name shd --gpu $gpu --data_folder SHD --nb_epochs 30 --log_tofile true --trials 3 --nb_layers 3  --pdrop 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]; then # reference - batchnorm
  python run_exp.py --model_type RLIF --dataset_name shd --gpu $gpu --data_folder SHD --nb_epochs 30 --log_tofile true --trials 3 --nb_layers 3  --normalization none
elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]; then # reference - dropout - batchnorm
  python run_exp.py --model_type RLIF --dataset_name shd --gpu $gpu --data_folder SHD --nb_epochs 30 --log_tofile true --trials 3 --nb_layers 3 --pdrop 0 --normalization none  
elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]; then # reference @ 2 layers
  python run_exp.py --model_type RLIF --dataset_name shd --gpu $gpu --data_folder SHD --nb_epochs 30 --log_tofile true --trials 3 --nb_layers 2
elif [[ $SLURM_ARRAY_TASK_ID -eq 6 ]]; then # reference @ 2 layers - dropout - batchnorm
  python run_exp.py --model_type RLIF --dataset_name shd --gpu $gpu --data_folder SHD --nb_epochs 30 --log_tofile true --trials 3 --nb_layers 2 --pdrop 0 --normalization none
elif [[ $SLURM_ARRAY_TASK_ID -eq 7 ]]; then # reference + one-spike-only
  python run_exp.py --model_type RLIF --dataset_name shd --gpu $gpu --data_folder SHD --nb_epochs 50 --log_tofile true --trials 3 --nb_layers 3 --balance true
elif [[ $SLURM_ARRAY_TASK_ID -eq 8 ]]; then # reference @ 2 layers - dropout - batchnorm + one-spike-only
  python run_exp.py --model_type RLIF --dataset_name shd --gpu $gpu --data_folder SHD --nb_epochs 50 --log_tofile true --trials 3 --nb_layers 2 --pdrop 0 --normalization none --balance true
elif [[ $SLURM_ARRAY_TASK_ID -eq 9 ]]; then # reference + one-spike-only + substep 5
  python run_exp.py --model_type RLIF --dataset_name shd --gpu $gpu --data_folder SHD --nb_epochs 50 --log_tofile true --trials 3 --nb_layers 3 --balance true --substeps 5
elif [[ $SLURM_ARRAY_TASK_ID -eq 10 ]]; then # reference @ 2 layers - dropout - batchnorm + one-spike-only + substep 5
  python run_exp.py --model_type RLIF --dataset_name shd --gpu $gpu --data_folder SHD --nb_epochs 50 --log_tofile true --trials 3 --nb_layers 2 --pdrop 0 --normalization none --balance true --substeps 5
elif [[ $SLURM_ARRAY_TASK_ID -eq 11 ]]; then # reference + one-spike-only + substep 10
  python run_exp.py --model_type RLIF --dataset_name shd --gpu $gpu --data_folder SHD --nb_epochs 50 --log_tofile true --trials 3 --nb_layers 3 --balance true --substeps 10
elif [[ $SLURM_ARRAY_TASK_ID -eq 12 ]]; then # reference @ 2 layers - dropout - batchnorm + one-spike-only + substep 10
  python run_exp.py --model_type RLIF --dataset_name shd --gpu $gpu --data_folder SHD --nb_epochs 50 --log_tofile true --trials 3 --nb_layers 2 --pdrop 0 --normalization none --balance true --substeps 10
elif [[ $SLURM_ARRAY_TASK_ID -eq 13 ]]; then # reference + one-spike-only + substep 20
  python run_exp.py --model_type RLIF --dataset_name shd --gpu $gpu --data_folder SHD --nb_epochs 50 --log_tofile true --trials 3 --nb_layers 3 --balance true --substeps 20
elif [[ $SLURM_ARRAY_TASK_ID -eq 14 ]]; then # reference @ 2 layers - dropout - batchnorm + one-spike-only + substep 20
  python run_exp.py --model_type RLIF --dataset_name shd --gpu $gpu --data_folder SHD --nb_epochs 50 --log_tofile true --trials 3 --nb_layers 2 --pdrop 0 --normalization none --balance true --substeps 20
elif [[ $SLURM_ARRAY_TASK_ID -eq 15 ]]; then # reference + one-spike-only + substep 50
  python run_exp.py --model_type RLIF --dataset_name shd --gpu $gpu --data_folder SHD --nb_epochs 50 --log_tofile true --trials 3 --nb_layers 3 --balance true --substeps 50
elif [[ $SLURM_ARRAY_TASK_ID -eq 16 ]]; then # reference @ 2 layers - dropout - batchnorm + one-spike-only + substep 50
  python run_exp.py --model_type RLIF --dataset_name shd --gpu $gpu --data_folder SHD --nb_epochs 50 --log_tofile true --trials 3 --nb_layers 2 --pdrop 0 --normalization none --balance true --substeps 50
fi

