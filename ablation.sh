#!/bin/bash

### example cluster options (refer to https://doc.itc.rwth-aachen.de/display/CC/Using+the+SLURM+Batch+System)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=baseline
#SBATCH --output=baseline.%J.txt
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1

### custom setup for python experiments
## setup conda
. $HOME/miniconda3/etc/profile.d/conda.sh
export PATH=$HOME/miniconda3/bin:$PATH
eval "$(conda shell.bash hook)"
conda activate beep

cd $HOME/Projects/sparch

echo "TASK: $SLURM_ARRAY_TASK_ID"
if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then # multi spike
  python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --new-exp-folder baseline_multispiker4 --trials 3 --plot --balance-metric lowpass --gpu 0 --repeat 4
elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then # multi spike
  python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --new-exp-folder baseline_multispiker8 --trials 3 --plot --balance-metric lowpass --gpu 0 --repeat 8  
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]; then # single spike
  python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --new-exp-folder baseline_singlespike --single-spike --repeat 4 --trials 3 --plot --balance-metric lowpass --gpu 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]; then # lsm
  python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --single-spike --repeat 4 --trials 3 --fix-w-in --fix-w-rec --fix-tau-out --fix-tau-rec --balance --plot --new-exp-folder lsm --balance-metric lowpass --gpu 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]; then # janek best 0
  python main.py --V-scale 0.2156983553211 --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0 --normalization none --track-balance --repeat 4 --batch-size 30 --single-spike --dataset-scale 200 --balance --fix-tau-out --fix-w-rec --new-exp-folder best0 --n-epochs 10 --trials 3 --balance-metric lowpass --plot --gpu 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]; then # janek best 1
  python main.py --V-scale 0.1261310791166 --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0 --normalization none --track-balance --repeat 4 --batch-size 30 --single-spike --dataset-scale 200 --balance --new-exp-folder best1 --n-epochs 10 --trials 3 --balance-metric lowpass --plot --gpu 0
fi

