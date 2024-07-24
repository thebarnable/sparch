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
fi

