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
  python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --new-exp-folder baseline_multispike --trials 3 --plot --balance-metric lowpass --gpu 0 --repeat 4
elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then # multi spike
  python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --new-exp-folder baseline_multispiker8 --trials 3 --plot --balance-metric lowpass --gpu 0 --repeat 8  
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]; then # single spike
  python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --new-exp-folder baseline_singlespike --single-spike --repeat 4 --trials 3 --plot --balance-metric lowpass --gpu 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]; then # lsm
  python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --single-spike --repeat 4 --trials 3 --fix-w-in --fix-w-rec --fix-tau-out --fix-tau-rec --balance --plot --new-exp-folder lsm --balance-metric lowpass --gpu 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]; then # remix
  python main.py --V-scale 0.111528 --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0 --normalization none --track-balance --repeat 4 --batch-size 30 --single-spike --dataset-scale 200 --balance --fix-w-rec --new-exp-folder remix --n-epochs 10 --trials 3 --balance-metric lowpass --plot --gpu 0 --mu 0.000264 --nu 0.000390
elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]; then # refit
  python main.py --V-scale 0.000013 --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0 --normalization none --track-balance --repeat 4 --batch-size 30 --single-spike --dataset-scale 200 --balance --fix-w-in --new-exp-folder refit --n-epochs 10 --trials 3 --balance-metric lowpass --plot --gpu 0 --mu 0.001588 --nu 0.000489 --balance-refit
elif [[ $SLURM_ARRAY_TASK_ID -eq 6 ]]; then # cuba
  python main.py --V-scale 0.294403 --V-slow-scale 0.187194 --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0 --normalization none --track-balance --repeat 4 --batch-size 30 --single-spike --dataset-scale 200 --balance --fix-w-in --new-exp-folder cuba --n-epochs 10 --trials 3 --balance-metric lowpass --plot --gpu 0 --mu 0.000162 --nu 0.003723 --slow-dynamics
elif [[ $SLURM_ARRAY_TASK_ID -eq 7 ]]; then # cuba+refit
  python main.py --V-scale 0.001121 --V-slow-scale 0.000012 --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0 --normalization none --track-balance --repeat 4 --batch-size 30 --single-spike --dataset-scale 200 --balance --fix-w-in --fix-tau-out --new-exp-folder cuba_refit --n-epochs 10 --trials 3 --balance-metric lowpass --plot --gpu 0 --mu 0.002899 --nu 0.000188 --slow-dynamics --balance-refit
elif [[ $SLURM_ARRAY_TASK_ID -eq 9 ]]; then # noise test
  for quant in 4 5 6 7 8 9 10 11 12; do
    echo "running quantization $quant"
    python main.py --V-scale 0.294403 --V-slow-scale 0.187194 --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0 --normalization none --track-balance --repeat 4 --batch-size 30 --single-spike --dataset-scale 200 --balance --fix-w-in --new-exp-folder quant$quant --n-epochs 10 --trials 3 --balance-metric lowpass --plot --gpu 0 --mu 0.000162 --nu 0.003723 --slow-dynamics --quantize $quant.1
  done
elif [[ $SLURM_ARRAY_TASK_ID -eq 8 ]]; then # cuba reasonable
  python main.py --V-scale 0.294403 --V-slow-scale 0.187194 --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0 --normalization none --track-balance --repeat 4 --batch-size 30 --single-spike --dataset-scale 200 --balance --fix-w-in --new-exp-folder cuba_fixed --n-epochs 10 --trials 3 --balance-metric lowpass --plot --gpu 0 --mu 0.000162 --nu 0.003723 --slow-dynamics --fix-w-rec
elif [[ $SLURM_ARRAY_TASK_ID -eq 10 ]]; then # lsm train tau out
  python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --single-spike --repeat 4 --trials 3 --fix-w-in --fix-w-rec --fix-tau-rec --balance --plot --new-exp-folder lsm_tests/train_tau_out --balance-metric lowpass --gpu 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 11 ]]; then # lsm train tau rec
  python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --single-spike --repeat 4 --trials 3 --fix-w-in --fix-w-rec --fix-tau-out --balance --plot --new-exp-folder lsm_tests/train_tau_rec --balance-metric lowpass --gpu 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 12 ]]; then # lsm train tau rec tau out
  python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --single-spike --repeat 4 --trials 3 --fix-w-in --fix-w-rec --balance --plot --new-exp-folder lsm_tests/train_taurec_tauout --balance-metric lowpass --gpu 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 13 ]]; then # lsm train win
  python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --single-spike --repeat 4 --trials 3 --fix-w-rec --fix-tau-out --fix-tau-rec --balance --plot --new-exp-folder lsm_tests/train_win --balance-metric lowpass --gpu 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 14 ]]; then # lsm train wrec
  python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --single-spike --repeat 4 --trials 3 --fix-w-in --fix-tau-out --fix-tau-rec --balance --plot --new-exp-folder lsm_tests/train_wrec --balance-metric lowpass --gpu 0
elif [[ $SLURM_ARRAY_TASK_ID -eq 15 ]]; then # lsm train wrec win
  python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --single-spike --repeat 4 --trials 3 --fix-tau-out --fix-tau-rec --balance --plot --new-exp-folder lsm_tests/train_wrec_win --balance-metric lowpass --gpu 0  
elif [[ $SLURM_ARRAY_TASK_ID -eq 16 ]]; then # lsm train all
  python main.py --model RLIF --dataset cue --n-layer 1 --neurons 100 --dropout 0.0 --normalization none --track-balance --batch-size 30 --dataset-scale 200 --n-epochs 10 --single-spike --repeat 4 --trials 3 --balance --plot --new-exp-folder lsm_tests/train_all --balance-metric lowpass --gpu 0
fi
